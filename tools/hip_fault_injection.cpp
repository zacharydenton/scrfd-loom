// Test-only interposer: fail the second launch after real GPU work was queued.
// Real synchronization always finishes before simulating a recovery error.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
namespace {
int remaining = 0, syncs = 0, launches = 0;
bool fail_drain = false;
}
extern "C" void loom_test_arm(int drain_error) {
    remaining = 2;
    syncs = launches = 0;
    fail_drain = drain_error != 0;
}
extern "C" int loom_test_count(int kind) { return kind == 0 ? syncs : launches; }
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz, unsigned shared, hipStream_t stream, void **params, void **extra) {
    ++launches;
    if (remaining > 0 && --remaining == 0) return hipErrorInvalidValue;
    static auto real = reinterpret_cast<decltype(&hipModuleLaunchKernel)>(dlsym(RTLD_NEXT, "hipModuleLaunchKernel"));
    return real(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}
extern "C" hipError_t hipDeviceSynchronize() {
    ++syncs;
    static auto real = reinterpret_cast<decltype(&hipDeviceSynchronize)>(dlsym(RTLD_NEXT, "hipDeviceSynchronize"));
    hipError_t result = real();
    if (fail_drain) { fail_drain = false; return hipErrorUnknown; }
    return result;
}
