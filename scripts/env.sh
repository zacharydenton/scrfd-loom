# Source this from Bash. See docs/building.md for the pinned toolchain build.
export HRX_BUILD="${HRX_BUILD:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build/hrx-system}"
export LOOM_TOOLS="${LOOM_TOOLS:-$HRX_BUILD/loom/src/loom/tools}"
export LOOM_COMPILE="${LOOM_COMPILE:-$LOOM_TOOLS/loom-compile/loom-compile}"
export LOOM_FORMAT="${LOOM_FORMAT:-$LOOM_TOOLS/loom-format/loom-format}"
export LOOM_CHECK="${LOOM_CHECK:-$LOOM_TOOLS/loom-check/loom-check}"
export IREE_TEST_LOOM="${IREE_TEST_LOOM:-$LOOM_TOOLS/iree-test-loom/iree-test-loom}"
export IREE_BENCHMARK_LOOM="${IREE_BENCHMARK_LOOM:-$LOOM_TOOLS/iree-benchmark-loom/iree-benchmark-loom}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIPCC="${HIPCC:-$ROCM_PATH/bin/hipcc}"
# Use the system runtime unless the caller explicitly selects another one.
if [ -n "${SCRFD_LOOM_RUNTIME_PATH:-}" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$SCRFD_LOOM_RUNTIME_PATH:"*) ;;
    *) export LD_LIBRARY_PATH="$SCRFD_LOOM_RUNTIME_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi
export LOOM_TARGET="${LOOM_TARGET:-gfx1151}"
