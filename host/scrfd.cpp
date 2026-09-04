// det_10g (SCRFD-10GF) forward pass on gfx1151: a resident session that walks the
// launch table tools/gen_launch_table.py generated from the ONNX graph.
//
// Nothing about the network is written here by hand. Each entry of the table
// names a kernel, its configuration, and the buffers it reads and writes; this
// file only knows how to build the kernarg block for each kernel kind.
//
// Build:  hipcc -O2 -Wall -Werror -o host/scrfd host/scrfd.cpp            (CLI)
//         hipcc -O2 -Wall -Werror -shared -fPIC -o host/libscrfd.so host/scrfd.cpp
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "scrfd.h"

namespace {

constexpr int SIZE = 640;
constexpr int MAX_BATCH = 64;
constexpr int THREADS = 256;
constexpr int TILE = 64;

struct scrfd_launch {
    int kind, variant, kernel;
    const char *name, *stage;
    int h, w, stride, cin_pad, cin_stride, k_size, n_size, ho, wo;
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
    switch (l.kind) {
    case 0: return "scrfd_nchw_to_nhwc_f16";
    case 1: return conv[l.variant];
    case 2: return l.variant == 4 ? "scrfd_matmul_add_resized_f16_wmma" : "dinov3_matmul_bias_f16_wmma_af16_cf16";
    default: return "scrfd_pool2_f16";
    }
}

class Session {
public:
    Session(const std::string &weights_dir, const std::string &kernels_dir, int max_batch)
        : max_batch_(max_batch) {
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
        HIP_CHECK(hipMalloc(&input_, size_t(max_batch) * 3 * SIZE * SIZE * sizeof(float)));
    }

    ~Session() {
        // Best-effort teardown; a destructor must not throw.
        for (auto *b : buffers_) (void)hipFree(b);
        (void)hipFree(input_); (void)hipFree(weights16_); (void)hipFree(weights32_);
        for (auto &k : kernels_) if (k.module) (void)hipModuleUnload(k.module);
    }

    int max_batch() const { return max_batch_; }
    Profiler profiler;

    static size_t head_elements(int level, int batch) {
        const auto &h = scrfd_heads[level];
        return size_t(batch) * h.h * h.w * 64;
    }

    void run(const float *input, size_t input_elements, int batch,
             uint16_t *heads[3], const size_t head_elems[3]) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (batch < 1 || batch > max_batch_)
            throw std::invalid_argument("batch must be 1.." + std::to_string(max_batch_));
        const size_t want = size_t(batch) * 3 * SIZE * SIZE;
        if (input_elements != want)
            throw std::invalid_argument("input has " + std::to_string(input_elements) + " elements, expected " +
                                        std::to_string(want) + " for batch " + std::to_string(batch));
        for (int lv = 0; lv < 3; ++lv)
            if (head_elems[lv] != head_elements(lv, batch))
                throw std::invalid_argument("head output " + std::to_string(lv) + " has the wrong element count");
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)input_, (void *)input, want * sizeof(float)));
        forward(batch);
        HIP_CHECK(hipDeviceSynchronize());
        for (int lv = 0; lv < 3; ++lv)
            HIP_CHECK(hipMemcpyDtoH(heads[lv], (hipDeviceptr_t)buffers_[scrfd_heads[lv].buf],
                                    head_elems[lv] * sizeof(uint16_t)));
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
                gx = l.n_size / TILE; gy = (m + TILE - 1) / TILE;
            }
            void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size, HIP_LAUNCH_PARAM_END};
            HIP_CHECK(hipModuleLaunchKernel(kernels_[l.kernel].function, gx, gy, 1, THREADS, 1, 1, 0,
                                            nullptr, nullptr, config));
        }
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
    int max_batch_;
    std::mutex mutex_;
    std::vector<Kernel> kernels_;
    std::vector<void *> buffers_;
    void *input_ = nullptr, *weights16_ = nullptr, *weights32_ = nullptr;
    std::map<int, size_t> weight_off_, bias_off_;
};

void set_error(char *error, size_t capacity, const char *message) {
    if (error && capacity) { strncpy(error, message, capacity - 1); error[capacity - 1] = '\0'; }
}

}  // namespace

struct scrfd_session { Session value; scrfd_session(const char *w, const char *k, int b) : value(w, k, b) {} };

extern "C" uint32_t scrfd_abi_version(void) { return SCRFD_ABI_VERSION; }
extern "C" int scrfd_input_size(void) { return SIZE; }
extern "C" int scrfd_max_batch(const scrfd_session *s) { return s ? s->value.max_batch() : 0; }
extern "C" void scrfd_destroy(scrfd_session *s) { delete s; }

extern "C" int scrfd_create(const char *weights_dir, const char *kernels_dir, int max_batch,
                            scrfd_session **out, char *error, size_t error_capacity) {
    if (!out) return 2;
    *out = nullptr;
    if (!weights_dir || !kernels_dir) { set_error(error, error_capacity, "weights_dir and kernels_dir are required"); return 2; }
    try {
        *out = new scrfd_session(weights_dir, kernels_dir, max_batch);
        return 0;
    } catch (const std::exception &e) {
        set_error(error, error_capacity, e.what());
        return 1;
    }
}

extern "C" int scrfd_run(scrfd_session *s, const float *input, size_t input_elements, int batch,
                         uint16_t *head8, size_t n8, uint16_t *head16, size_t n16,
                         uint16_t *head32, size_t n32, char *error, size_t error_capacity) {
    if (!s || !input || !head8 || !head16 || !head32) { set_error(error, error_capacity, "null argument"); return 2; }
    try {
        uint16_t *heads[3] = {head8, head16, head32};
        const size_t elems[3] = {n8, n16, n32};
        s->value.run(input, input_elements, batch, heads, elems);
        return 0;
    } catch (const std::exception &e) {
        set_error(error, error_capacity, e.what());
        return 1;
    }
}

#ifndef SCRFD_NO_MAIN
// CLI: the same session driven from files, for validation and profiling.
//   host/scrfd --weights build/weights --kernels build/kernels --input blob.bin
//              [--batch N] [--repeat N] [--profile] [--output heads.bin]
// --input holds either one 3x640x640 f32 image, replicated across the batch, or
// exactly --batch of them. --output writes the three head tensors back to back.
int main(int argc, char **argv) {
    std::string weights_dir = "build/weights", kernels_dir = "build/kernels", input_path, output_path;
    int batch = 1, repeat = 1; bool profile = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(64); }
            return std::string(argv[++i]);
        };
        if (a == "--weights") weights_dir = next();
        else if (a == "--kernels") kernels_dir = next();
        else if (a == "--input") input_path = next();
        else if (a == "--output") output_path = next();
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--repeat") repeat = std::stoi(next());
        else if (a == "--profile") profile = true;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 64; }
    }
    if (batch < 1 || batch > MAX_BATCH) { fprintf(stderr, "--batch must be 1..%d\n", MAX_BATCH); return 64; }
    if (input_path.empty()) { fprintf(stderr, "--input is required (3x%dx%d f32 per image)\n", SIZE, SIZE); return 64; }
    const size_t per_image = size_t(3) * SIZE * SIZE;
    std::vector<char> raw;
    try { raw = read_file(input_path); } catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 64; }
    if (raw.size() % (per_image * 4) != 0) {
        fprintf(stderr, "%s is %zu bytes, not a multiple of one 3x%dx%d f32 image\n", input_path.c_str(), raw.size(), SIZE, SIZE);
        return 64;
    }
    const size_t supplied = raw.size() / (per_image * 4);
    if (supplied != 1 && supplied != size_t(batch)) {
        fprintf(stderr, "%s holds %zu images; expected 1 (replicated) or %d (--batch)\n", input_path.c_str(), supplied, batch);
        return 64;
    }
    std::vector<float> input(per_image * batch);
    for (int b = 0; b < batch; ++b)
        memcpy(input.data() + size_t(b) * per_image,
               raw.data() + (supplied == 1 ? 0 : size_t(b) * per_image * 4), per_image * 4);

    scrfd_session *session = nullptr;
    char error[512];
    if (scrfd_create(weights_dir.c_str(), kernels_dir.c_str(), batch, &session, error, sizeof error)) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    std::vector<std::vector<uint16_t>> heads(3);
    size_t elems[3];
    for (int lv = 0; lv < 3; ++lv) { elems[lv] = Session::head_elements(lv, batch); heads[lv].resize(elems[lv]); }
    auto run_once = [&]() {
        if (scrfd_run(session, input.data(), input.size(), batch, heads[0].data(), elems[0],
                      heads[1].data(), elems[1], heads[2].data(), elems[2], error, sizeof error)) {
            fprintf(stderr, "%s\n", error); exit(1);
        }
    };
    run_once();                                            // warm
    session->value.profiler.enabled = profile;
    session->value.profiler.stage_us.clear(); session->value.profiler.stage_calls.clear();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_once();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int images = repeat * batch;
    printf("{\"batch\": %d, \"images\": %d, \"total_ms\": %.3f, \"ms_per_image\": %.4f, \"img_per_s\": %.2f}\n",
           batch, images, ms, ms / images, images / (ms / 1000.0));
    if (profile) session->value.print_profile(repeat, batch);
    if (!output_path.empty()) {
        std::ofstream out(output_path, std::ios::binary);
        for (int lv = 0; lv < 3; ++lv)
            out.write(reinterpret_cast<const char *>(heads[lv].data()), elems[lv] * sizeof(uint16_t));
    }
    scrfd_destroy(session);
    return 0;
}
#endif
