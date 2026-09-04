// det_10g (SCRFD-10GF) forward pass on gfx1151: a resident session that walks the
// launch table tools/gen_launch_table.py generated from the ONNX graph, then
// decodes the fused head tensors to candidate detections on the host.
//
// Nothing about the network is written here by hand. Each entry of the table
// names a kernel, its configuration, and the buffers it reads and writes; this
// file only knows how to build the kernarg block for each kernel kind, and
// insightface's anchor arithmetic (distance2bbox / distance2kps).
//
// Build: ./scripts/build_host.sh  (host/scrfd CLI and build/libscrfd.so)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "scrfd.h"

namespace {

constexpr int SIZE = 640;
constexpr int CHANNELS = 3;                 // BGR bytes per input pixel
constexpr int MAX_BATCH = 64;
constexpr int THREADS = 256;
constexpr int TILE = 64;
constexpr int LEVELS = 3;
constexpr int STRIDES[LEVELS] = {8, 16, 32};
constexpr int HEAD_WIDTH = 64;              // fused head row: [score(2) | box(8) | kps(20) | zero pad]
constexpr int ANCHORS = 2;

struct scrfd_launch {
    int kind, variant, kernel;
    const char *name, *stage;
    int h, w, stride, cin_pad, cin_stride, k_size, n_size, ho, wo;
    int tile;                       // N tile of a 3x3 conv: 64, or 128 (tile_for in tools/export_weights.py)
    int src_buf, dst_buf, extra_buf;
};
struct scrfd_head { const char *name; int buf, h, w; };

#include "graph_table.inc"

#define HIP_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(e_)); } while (0)

struct Span { size_t offset, count; };

std::map<std::string, Span> read_manifest(const std::string &path) {
    std::map<std::string, Span> spans;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string name; size_t offset, count;
    while (in >> name >> offset >> count) spans[name] = {offset, count};
    return spans;
}

std::vector<char> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<char> buffer(in.tellg());
    in.seekg(0);
    in.read(buffer.data(), buffer.size());
    return buffer;
}

// Every span inside the blob, both multiplications overflow-checked.
void check_spans(const std::map<std::string, Span> &spans, size_t blob_bytes,
                 size_t element_bytes, const char *what) {
    for (const auto &e : spans) {
        size_t begin, extent, end;
        if (__builtin_mul_overflow(e.second.offset, element_bytes, &begin) ||
            __builtin_mul_overflow(e.second.count, element_bytes, &extent) ||
            __builtin_add_overflow(begin, extent, &end) || end > blob_bytes)
            throw std::runtime_error(std::string(what) + ": span '" + e.first + "' runs past the blob");
    }
}

// Every tensor a launch reads must exist with exactly the element count its
// kernel is compiled for: the kernels take bare pointers and never see counts.
size_t require(const std::map<std::string, Span> &spans, const std::string &name, size_t count,
               const char *what) {
    auto it = spans.find(name);
    if (it == spans.end()) throw std::runtime_error(std::string(what) + ": missing tensor '" + name + "'");
    if (it->second.count != count)
        throw std::runtime_error(std::string(what) + ": tensor '" + name + "' has " +
                                 std::to_string(it->second.count) + " elements, the kernel expects " +
                                 std::to_string(count));
    return it->second.offset;
}

struct Kernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    void load(const std::string &path, const char *symbol) {
        HIP_CHECK(hipModuleLoad(&module, path.c_str()));
        HIP_CHECK(hipModuleGetFunction(&function, module, symbol));
    }
};

// Loom's AMDGPU kernarg ABI: i32 scalars 4-byte aligned, pointers 8-byte aligned.
struct KernArgs {
    alignas(16) unsigned char bytes[128];
    size_t size = 0;
    void scalar_i32(int v) { size = (size + 3) & ~size_t(3); memcpy(bytes + size, &v, 4); size += 4; }
    void pointer(const void *p) { size = (size + 7) & ~size_t(7); memcpy(bytes + size, &p, 8); size += 8; }
};

struct Profiler {
    bool enabled = false;
    std::map<std::string, double> stage_us;
    std::map<std::string, int> stage_calls;
};

struct Stage {
    Profiler &p; std::string name; std::chrono::steady_clock::time_point start;
    Stage(Profiler &prof, const char *n) : p(prof), name(n) {
        if (p.enabled) { HIP_CHECK(hipDeviceSynchronize()); start = std::chrono::steady_clock::now(); }
    }
    ~Stage() {
        if (!p.enabled) return;
        (void)hipDeviceSynchronize();   // a destructor must not throw
        auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        p.stage_us[name] += us; p.stage_calls[name] += 1;
    }
};

const char *symbol_for(const scrfd_launch &l) {
    static const char *conv[] = {"scrfd_conv3x3_f16_wmma", "scrfd_conv3x3_f16_wmma_relu",
                                 "scrfd_conv3x3_f16_wmma_add", "scrfd_conv3x3_f16_wmma_relu_add"};
    static const char *wide[] = {"scrfd_conv3x3_n128_f16_wmma", "scrfd_conv3x3_n128_f16_wmma_relu",
                                 "scrfd_conv3x3_n128_f16_wmma_add", "scrfd_conv3x3_n128_f16_wmma_relu_add"};
    switch (l.kind) {
    case 0: return "scrfd_hwc_u8_to_nhwc_f16";
    case 1: return l.tile == 128 ? wide[l.variant] : conv[l.variant];
    case 2: return l.variant == 4 ? "scrfd_matmul_add_resized_f16_wmma" : "dinov3_matmul_bias_f16_wmma_af16_cf16";
    default: return "scrfd_pool2_f16";
    }
}

static_assert(sizeof(_Float16) == 2, "the head tensors are IEEE half");
inline float half_to_float(uint16_t bits) {
    _Float16 h;
    memcpy(&h, &bits, sizeof h);
    return static_cast<float>(h);
}

class Session {
public:
    Session(const std::string &weights_dir, const std::string &kernels_dir, int max_batch)
        : max_batch_(max_batch) {
        try {
            if (max_batch < 1 || max_batch > MAX_BATCH)
                throw std::invalid_argument("max_batch must be 1.." + std::to_string(MAX_BATCH));
            HIP_CHECK(hipInit(0));

            auto spans16 = read_manifest(weights_dir + "/manifest_f16.txt");
            auto spans32 = read_manifest(weights_dir + "/manifest.txt");
            auto blob16 = read_file(weights_dir + "/weights_f16.bin");
            auto blob32 = read_file(weights_dir + "/weights.bin");
            check_spans(spans16, blob16.size(), 2, "manifest_f16.txt");
            check_spans(spans32, blob32.size(), 4, "manifest.txt");
            HIP_CHECK(hipMalloc(&weights16_, blob16.size()));
            HIP_CHECK(hipMalloc(&weights32_, blob32.size()));
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights16_, blob16.data(), blob16.size()));
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights32_, blob32.data(), blob32.size()));

            kernels_.resize(SCRFD_KERNEL_COUNT);
            std::vector<bool> loaded(SCRFD_KERNEL_COUNT, false);
            for (int i = 0; i < SCRFD_LAUNCH_COUNT; ++i) {
                const auto &l = scrfd_launches[i];
                if (!loaded[l.kernel]) {
                    kernels_[l.kernel].load(kernels_dir + "/" + scrfd_kernel_stems[l.kernel] + ".hsaco", symbol_for(l));
                    loaded[l.kernel] = true;
                }
                if (l.kind == 1 || l.kind == 2) {
                    weight_off_[i] = require(spans16, l.name, size_t(l.n_size) * l.k_size, "manifest_f16.txt");
                    bias_off_[i] = require(spans32, std::string(l.name) + "_b", size_t(l.n_size), "manifest.txt");
                }
            }

            buffers_.resize(SCRFD_BUFFER_COUNT);
            for (int b = 0; b < SCRFD_BUFFER_COUNT; ++b)
                HIP_CHECK(hipMalloc(&buffers_[b], scrfd_buffer_bytes[b] * max_batch));
            HIP_CHECK(hipMalloc(&input_, input_bytes(max_batch)));
            // The heads come back to pinned host memory: the copy is faster and the
            // decode reads it straight away.
            for (int lv = 0; lv < LEVELS; ++lv)
                HIP_CHECK(hipHostMalloc(&heads_host_[lv], head_elements(lv, max_batch) * sizeof(uint16_t),
                                        hipHostMallocDefault));
        } catch (...) {
            release();
            throw;
        }
    }

    ~Session() { release(); }

    int max_batch() const { return max_batch_; }
    Profiler profiler;

    static size_t input_bytes(int batch) { return size_t(batch) * SIZE * SIZE * CHANNELS; }
    static size_t head_elements(int level, int batch) {
        const auto &h = scrfd_heads[level];
        return size_t(batch) * h.h * h.w * HEAD_WIDTH;
    }
    // The last run's head tensor for a level, [batch*H*W][64] f16 (scores pre-sigmoid).
    const uint16_t *heads(int level) const { return heads_host_[level]; }

    void check_batch(int batch) const {
        if (batch < 1 || batch > max_batch_)
            throw std::invalid_argument("batch must be 1.." + std::to_string(max_batch_) +
                                        " (max_batch at creation), got " + std::to_string(batch));
    }

    void upload(const uint8_t *input, size_t bytes, int batch) {
        if (!input) throw std::invalid_argument("input must not be null");
        check_batch(batch);
        const size_t want = input_bytes(batch);
        if (bytes != want)
            throw std::invalid_argument("input has " + std::to_string(bytes) + " bytes; batch " +
                                        std::to_string(batch) + " requires exactly " + std::to_string(want) +
                                        " (" + std::to_string(SIZE) + "x" + std::to_string(SIZE) + " BGR uint8 per image)");
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)input_, (void *)input, want));
    }

    void forward(int batch) {
        for (int i = 0; i < SCRFD_LAUNCH_COUNT; ++i) {
            const auto &l = scrfd_launches[i];
            Stage stage(profiler, l.stage);
            KernArgs args;
            void *src = l.src_buf < 0 ? input_ : buffers_[l.src_buf];
            void *dst = buffers_[l.dst_buf];
            const int m = batch * l.ho * l.wo;
            unsigned gx, gy;
            if (l.kind == 0) {                       // convert: one workgroup per image row
                args.scalar_i32(batch * l.h);
                args.pointer(src); args.pointer(dst);
                gx = batch * l.h; gy = 1;
            } else if (l.kind == 3) {                // pool: one workgroup per output row
                args.scalar_i32(m);
                args.pointer(src); args.pointer(dst);
                gx = batch * l.ho; gy = 1;
            } else {                                 // conv3x3 / matmul: 64x64 tiles
                args.scalar_i32(m);
                args.pointer(src);
                args.pointer((char *)weights16_ + weight_off_[i] * 2);
                args.pointer((char *)weights32_ + bias_off_[i] * 4);
                args.pointer(dst);
                if (l.extra_buf >= 0) args.pointer(buffers_[l.extra_buf]);
                gx = l.n_size / (l.kind == 1 ? l.tile : TILE); gy = (m + TILE - 1) / TILE;   // M tile is always 64
            }
            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size, HIP_LAUNCH_PARAM_END};
            HIP_CHECK(hipModuleLaunchKernel(kernels_[l.kernel].function, gx, gy, 1, THREADS, 1, 1, 0,
                                            nullptr, nullptr, config));
        }
    }

    void synchronize() { HIP_CHECK(hipDeviceSynchronize()); }

    void download_heads(int batch) {
        Stage stage(profiler, "download heads");
        for (int lv = 0; lv < LEVELS; ++lv)
            HIP_CHECK(hipMemcpyDtoH(heads_host_[lv], (hipDeviceptr_t)buffers_[scrfd_heads[lv].buf],
                                    head_elements(lv, batch) * sizeof(uint16_t)));
    }

    // insightface's decode, per anchor: sigmoid the score, keep it if >= det_thresh,
    // distance2bbox / distance2kps around the anchor centre (x*stride, y*stride)
    // with the distances scaled by the stride. f32 throughout, as insightface's
    // NumPy path is. Returns the total number of candidates.
    size_t decode(int batch, float det_thresh, float *candidates, size_t max_candidates, int32_t *counts) const {
        Stage stage(const_cast<Profiler &>(profiler), "decode (host)");
        // sigmoid(x) >= t  <=>  x >= logit(t). Scan the raw logits against a slightly
        // relaxed bound and apply the exact f32 test only to the survivors, so the
        // exp runs for a few hundred anchors per image rather than 33600.
        const float logit_bound = std::log(det_thresh / (1.0f - det_thresh)) - 1e-3f;
        size_t total = 0;
        for (int b = 0; b < batch; ++b) {
            float *out = candidates + size_t(b) * max_candidates * SCRFD_CANDIDATE_FLOATS;
            size_t n = 0;
            for (int lv = 0; lv < LEVELS; ++lv) {
                const int stride = STRIDES[lv], side = SIZE / stride;
                const uint16_t *rows = heads_host_[lv] + size_t(b) * side * side * HEAD_WIDTH;
                for (int y = 0; y < side; ++y) {
                    for (int x = 0; x < side; ++x) {
                        const uint16_t *row = rows + (size_t(y) * side + x) * HEAD_WIDTH;
                        for (int a = 0; a < ANCHORS; ++a) {
                            const float logit = half_to_float(row[a]);
                            if (logit < logit_bound) continue;
                            const float score = 1.0f / (1.0f + std::exp(-logit));
                            if (!(score >= det_thresh)) continue;
                            if (n == max_candidates)
                                throw std::runtime_error("image " + std::to_string(b) + " has more than " +
                                                         std::to_string(max_candidates) +
                                                         " candidates above det_thresh; raise max_candidates");
                            const float cx = float(x * stride), cy = float(y * stride);
                            const uint16_t *box = row + 2 + 4 * a;
                            const uint16_t *kps = row + 10 + 10 * a;
                            float *c = out + n * SCRFD_CANDIDATE_FLOATS;
                            c[0] = cx - half_to_float(box[0]) * float(stride);
                            c[1] = cy - half_to_float(box[1]) * float(stride);
                            c[2] = cx + half_to_float(box[2]) * float(stride);
                            c[3] = cy + half_to_float(box[3]) * float(stride);
                            c[4] = score;
                            for (int k = 0; k < 5; ++k) {
                                c[5 + 2 * k] = cx + half_to_float(kps[2 * k]) * float(stride);
                                c[6 + 2 * k] = cy + half_to_float(kps[2 * k + 1]) * float(stride);
                            }
                            ++n;
                        }
                    }
                }
            }
            counts[b] = int32_t(n);
            total += n;
        }
        return total;
    }

    // One call of the ABI: validate everything before touching the GPU.
    size_t session_run(const uint8_t *input, size_t bytes, int batch, float det_thresh,
                       float *candidates, size_t candidates_elements, int32_t *counts, size_t counts_elements) {
        std::lock_guard<std::mutex> lock(mutex_);
        check_batch(batch);
        if (!candidates || !counts) throw std::invalid_argument("candidates and counts must not be null");
        if (!(det_thresh > 0.0f && det_thresh < 1.0f))
            throw std::invalid_argument("det_thresh must be strictly between 0 and 1");
        const size_t per_row = size_t(batch) * SCRFD_CANDIDATE_FLOATS;
        if (candidates_elements < per_row || candidates_elements % per_row != 0)
            throw std::invalid_argument("candidates has " + std::to_string(candidates_elements) +
                                        " f32 elements; batch " + std::to_string(batch) + " requires a positive multiple of " +
                                        std::to_string(per_row) + " (max_candidates * batch * " +
                                        std::to_string(SCRFD_CANDIDATE_FLOATS) + ")");
        if (counts_elements != size_t(batch))
            throw std::invalid_argument("counts has " + std::to_string(counts_elements) + " elements; batch " +
                                        std::to_string(batch) + " requires exactly " + std::to_string(batch));
        upload(input, bytes, batch);
        forward(batch);
        synchronize();
        download_heads(batch);
        return decode(batch, det_thresh, candidates, candidates_elements / per_row, counts);
    }

    void print_profile(int forwards, int batch) const {
        double total = 0;
        for (const auto &e : profiler.stage_us) total += e.second;
        std::vector<std::pair<double, std::string>> rows;
        for (const auto &e : profiler.stage_us) rows.push_back({e.second, e.first});
        std::sort(rows.rbegin(), rows.rend());
        printf("stage breakdown over %d forward pass(es), %d image(s):\n", forwards, forwards * batch);
        for (const auto &r : rows)
            printf("  %-24s %9.3f ms  %5.1f%%  (%d launches)\n", r.second.c_str(), r.first / 1000.0,
                   total ? 100.0 * r.first / total : 0.0, profiler.stage_calls.at(r.second));
        printf("  %-24s %9.3f ms\n", "total", total / 1000.0);
    }

private:
    void release() noexcept {
        // Construction can fail after any individual allocation or module load.
        // Clear every handle as it is released so this is also safe for normal
        // destruction and for partially initialized vectors.
        for (auto *&b : buffers_) {
            if (b) (void)hipFree(b);
            b = nullptr;
        }
        for (auto *&h : heads_host_) {
            if (h) (void)hipHostFree(h);
            h = nullptr;
        }
        if (input_) (void)hipFree(input_);
        if (weights16_) (void)hipFree(weights16_);
        if (weights32_) (void)hipFree(weights32_);
        input_ = weights16_ = weights32_ = nullptr;
        for (auto &k : kernels_) {
            if (k.module) (void)hipModuleUnload(k.module);
            k.module = nullptr;
            k.function = nullptr;
        }
    }

    int max_batch_;
    std::mutex mutex_;
    std::vector<Kernel> kernels_;
    std::vector<void *> buffers_;
    void *input_ = nullptr, *weights16_ = nullptr, *weights32_ = nullptr;
    uint16_t *heads_host_[LEVELS] = {nullptr, nullptr, nullptr};
    std::map<int, size_t> weight_off_, bias_off_;
};

void write_error(char *error, size_t capacity, const char *message) noexcept {
    if (!error || capacity == 0) return;
    std::snprintf(error, capacity, "%s", message ? message : "unknown error");
}

void clear_error(char *error, size_t capacity) noexcept {
    if (error && capacity) error[0] = '\0';
}

}  // namespace

struct scrfd_session {
    Session value;
    scrfd_session(const char *w, const char *k, int b) : value(w, k, b) {}
};

extern "C" uint32_t scrfd_abi_version(void) { return SCRFD_ABI_VERSION; }
extern "C" int scrfd_input_size(void) { return SIZE; }
extern "C" int scrfd_max_batch(const scrfd_session *s) { return s ? s->value.max_batch() : 0; }
extern "C" void scrfd_destroy(scrfd_session *s) { delete s; }

extern "C" int scrfd_create(const char *weights_dir, const char *kernels_dir, int max_batch,
                            scrfd_session **out_session, char *error, size_t error_capacity) {
    clear_error(error, error_capacity);
    if (!out_session) { write_error(error, error_capacity, "out_session must not be null"); return SCRFD_INVALID_ARGUMENT; }
    *out_session = nullptr;
    try {
        if (!weights_dir || !kernels_dir) throw std::invalid_argument("weights_dir and kernels_dir are required");
        *out_session = new scrfd_session(weights_dir, kernels_dir, max_batch);
        return SCRFD_OK;
    } catch (const std::invalid_argument &e) { write_error(error, error_capacity, e.what()); return SCRFD_INVALID_ARGUMENT; }
      catch (const std::exception &e)        { write_error(error, error_capacity, e.what()); return SCRFD_ERROR; }
      catch (...)                            { write_error(error, error_capacity, "unknown C++ exception"); return SCRFD_ERROR; }
}

extern "C" int scrfd_run(scrfd_session *s, const uint8_t *input, size_t input_bytes, int batch, float det_thresh,
                         float *candidates, size_t candidates_elements, int32_t *counts, size_t counts_elements,
                         char *error, size_t error_capacity) {
    clear_error(error, error_capacity);
    try {
        if (!s) throw std::invalid_argument("session must not be null");
        s->value.session_run(input, input_bytes, batch, det_thresh, candidates, candidates_elements, counts, counts_elements);
        return SCRFD_OK;
    } catch (const std::invalid_argument &e) { write_error(error, error_capacity, e.what()); return SCRFD_INVALID_ARGUMENT; }
      catch (const std::exception &e)        { write_error(error, error_capacity, e.what()); return SCRFD_ERROR; }
      catch (...)                            { write_error(error, error_capacity, "unknown C++ exception"); return SCRFD_ERROR; }
}

#ifndef SCRFD_LIBRARY
namespace {

int parse_integer(const std::string &option, const std::string &text) {
    errno = 0;
    char *end = nullptr;
    long value = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || value < INT_MIN || value > INT_MAX)
        throw std::invalid_argument(option + " must be an integer, got '" + text + "'");
    return static_cast<int>(value);
}

float parse_float(const std::string &option, const std::string &text) {
    errno = 0;
    char *end = nullptr;
    float value = std::strtof(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || !std::isfinite(value))
        throw std::invalid_argument(option + " must be a finite number, got '" + text + "'");
    return value;
}

// CLI: the same session driven from files, for validation and profiling.
//   host/scrfd --weights build/weights --kernels build/kernels --input images.bin
//              [--batch N] [--repeat N] [--thresh T] [--profile] [--output heads.bin]
// --input holds either one 640x640x3 BGR uint8 image, replicated across the
// batch, or exactly --batch of them. Each timed run is exactly what the ABI does
// (upload, forward, download, decode). --output writes the three raw head
// tensors back to back, for the comparison against onnxruntime.
int cli_main(int argc, char **argv) {
    std::string weights_dir = "build/weights", kernels_dir = "build/kernels", input_path, output_path;
    int batch = 1, repeat = 1; bool profile = false; float det_thresh = 0.5f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--weights") weights_dir = next();
        else if (a == "--kernels") kernels_dir = next();
        else if (a == "--input") input_path = next();
        else if (a == "--output") output_path = next();
        else if (a == "--batch") batch = parse_integer(a, next());
        else if (a == "--repeat") repeat = parse_integer(a, next());
        else if (a == "--thresh") det_thresh = parse_float(a, next());
        else if (a == "--profile") profile = true;
        else throw std::invalid_argument("unknown option " + a);
    }
    if (batch < 1 || batch > MAX_BATCH) throw std::invalid_argument("--batch must be 1.." + std::to_string(MAX_BATCH));
    if (repeat < 1) throw std::invalid_argument("--repeat must be at least 1");
    if (input_path.empty())
        throw std::invalid_argument("--input is required (" + std::to_string(SIZE) + "x" + std::to_string(SIZE) + "x3 BGR uint8 per image)");
    const size_t per_image = Session::input_bytes(1);
    std::vector<char> raw;
    try { raw = read_file(input_path); } catch (const std::exception &e) { throw std::invalid_argument(e.what()); }
    if (raw.empty() || raw.size() % per_image != 0)
        throw std::invalid_argument(input_path + " is " + std::to_string(raw.size()) + " bytes, not a multiple of one " +
                                    std::to_string(SIZE) + "x" + std::to_string(SIZE) + "x3 uint8 image");
    const size_t supplied = raw.size() / per_image;
    if (supplied != 1 && supplied != size_t(batch))
        throw std::invalid_argument(input_path + " holds " + std::to_string(supplied) + " images; expected 1 (replicated) or " +
                                    std::to_string(batch) + " (--batch)");
    std::vector<uint8_t> input(per_image * batch);
    for (int b = 0; b < batch; ++b)
        memcpy(input.data() + size_t(b) * per_image, raw.data() + (supplied == 1 ? 0 : size_t(b) * per_image), per_image);

    Session session(weights_dir, kernels_dir, batch);
    const size_t max_candidates = 4096;
    std::vector<float> candidates(size_t(batch) * max_candidates * SCRFD_CANDIDATE_FLOATS);
    std::vector<int32_t> counts(batch);
    size_t total = 0;
    auto run_once = [&]() {
        total = session.session_run(input.data(), input.size(), batch, det_thresh,
                                    candidates.data(), candidates.size(), counts.data(), counts.size());
    };
    run_once();                                            // warm
    session.profiler.enabled = profile;
    session.profiler.stage_us.clear(); session.profiler.stage_calls.clear();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_once();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int images = repeat * batch;
    printf("{\"batch\": %d, \"images\": %d, \"total_ms\": %.3f, \"ms_per_image\": %.4f, \"img_per_s\": %.2f, "
           "\"candidates\": %zu}\n", batch, images, ms, ms / images, images / (ms / 1000.0), total);
    if (profile) session.print_profile(repeat, batch);
    if (!output_path.empty()) {
        std::ofstream out(output_path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write " + output_path);
        for (int lv = 0; lv < LEVELS; ++lv)
            out.write(reinterpret_cast<const char *>(session.heads(lv)),
                      static_cast<std::streamsize>(Session::head_elements(lv, batch) * sizeof(uint16_t)));
        if (!out) throw std::runtime_error("cannot write all of " + output_path);
    }
    return SCRFD_OK;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return cli_main(argc, argv);
    } catch (const std::invalid_argument &e) { fprintf(stderr, "%s\n", e.what()); return SCRFD_INVALID_ARGUMENT; }
      catch (const std::exception &e)        { fprintf(stderr, "%s\n", e.what()); return SCRFD_ERROR; }
      catch (...)                            { fprintf(stderr, "unknown C++ exception\n"); return SCRFD_ERROR; }
}
#endif
