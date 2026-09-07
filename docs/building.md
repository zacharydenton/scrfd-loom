# Building scrfd-loom

The source checkout builds the native library and GPU kernels. Python wheels
contain only the loader and decode code; weights and native assets remain
external.

## Requirements

- Linux x86-64 with an AMD Radeon 8060S (gfx1151), an amdgpu kernel driver, and
  access to `/dev/kfd` and the render node under `/dev/dri`.
- ROCm with HIP headers, `hipcc`, and ROCm clang. The release checks use HIP
  7.2.53211 and clang 22 from `/opt/rocm`, with the system HIP runtime. Follow
  [AMD's installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
  for your distribution and hardware.
- Python 3.11 or newer, Git, CMake 3.26 or newer, Ninja, a C/C++ toolchain,
  `curl`, and `unzip`.
- The `det_10g.onnx` model from InsightFace's `buffalo_l` pack.

## Loom compiler

Run these commands from the scrfd-loom checkout. The pinned public revision is
[`c9855b47e96e7eb1cbb5b81b1de973762982ae95`](https://github.com/ROCm/hrx-system/commit/c9855b47e96e7eb1cbb5b81b1de973762982ae95).
No local HRX patches or HRX HIP compatibility library are needed. CMake fetches
the source dependencies pinned by that revision, so configuration needs network
access.

```bash
mkdir -p build
git clone https://github.com/ROCm/hrx-system.git build/hrx-source
git -C build/hrx-source checkout --detach c9855b47e96e7eb1cbb5b81b1de973762982ae95

export ROCM_PATH=/opt/rocm
cmake -S build/hrx-source -B build/hrx-system -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DIREE_ROCM_PATH="$ROCM_PATH" \
  -DIREE_ROCM_DEPENDENCY_MODE=pinned \
  -DIREE_BUILD_TESTS=OFF \
  -DLIBHRX_BUILD=OFF \
  -DLOOM_BUILD=ON \
  -DIREE_HAL_DRIVER_AMDGPU=ON
cmake --build build/hrx-system --target loom-compile loom-format --parallel 8
source scripts/env.sh
```

The compiler and formatter are the only Loom executables used by this project's
test suite. GPU tests run through the HIP host programs built here.

## Model

InsightFace's model terms are separate from this project's Apache-2.0 license.
The pack is distributed for non-commercial research; other uses require
appropriate licensing. See [InsightFace's policy](https://github.com/deepinsight/insightface#license).

Download the model from the upstream release and verify the file used by the
exporter and reference tests:

```bash
mkdir -p build/downloads "$HOME/.insightface/models/buffalo_l"
curl --fail --location --retry 3 \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
  --output build/downloads/buffalo_l.zip
unzip -j build/downloads/buffalo_l.zip det_10g.onnx \
  -d "$HOME/.insightface/models/buffalo_l"
export SCRFD_ONNX="$HOME/.insightface/models/buffalo_l/det_10g.onnx"
printf '%s  %s\n' \
  5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91 \
  "$SCRFD_ONNX" | sha256sum --check
```

If you already have the model, set `SCRFD_ONNX` to its absolute path and run the
checksum command. The exporter reads that file and writes `build/weights`.

## Build and test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
source scripts/env.sh
python tools/export_weights.py
python tools/gen_launch_table.py
./scripts/build_kernels.sh
./scripts/build_host.sh
./scripts/test.sh
```

The suite checks formatting and generated files, graph invariants, GPU kernels,
native error recovery, end-to-end detections and the Python API. It builds the
assets again as part of verification. `--quick` skips the final
reference/end-to-end/API group, but still requires the GPU and model.
The first test run downloads InsightFace's sample image into `build/images`.
The standard CPU ONNX Runtime package is sufficient; MIGraphX is required only
for the benchmark comparison.

## Paths and runtime selection

Set build overrides **before** sourcing `scripts/env.sh` in a fresh shell.
The script preserves explicit tool paths and exports them for Python tests.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HRX_BUILD` | `<checkout>/build/hrx-system` | Existing Loom CMake build |
| `LOOM_TOOLS` | `$HRX_BUILD/loom/src/loom/tools` | Loom tools directory |
| `LOOM_COMPILE`, `LOOM_FORMAT` | Executables under `LOOM_TOOLS` | Individual tool overrides |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation |
| `HIPCC` | `$ROCM_PATH/bin/hipcc` | Host compiler |
| `LOOM_TARGET` | `gfx1151` | Kernel target; other chips are unvalidated |
| `SCRFD_ONNX` | `~/.insightface/models/buffalo_l/det_10g.onnx` | Export/reference model |
| `SCRFD_LOOM_RUNTIME_PATH` | Unset | Optional runtime library directory, prepended to `LD_LIBRARY_PATH` |
| `SCRFD_LOOM_WEIGHTS` | `<checkout>/build/weights` | Exported weights for the Python API |
| `SCRFD_LOOM_KERNELS` | `<checkout>/build/kernels` | HSACOs for the Python API |
| `SCRFD_LOOM_LIBRARY` | `<checkout>/build/libscrfd.so` | Native library for the Python API |

The default uses the system HIP runtime and leaves `LD_LIBRARY_PATH` unchanged.
There is no dependency on an extracted toolbox runtime. For a ROCm installation
outside the loader's search path, set `SCRFD_LOOM_RUNTIME_PATH` to its library
directory before sourcing the environment. Use a consistent ROCm installation
when loading the native library and an ONNX Runtime MIGraphX build in one process.

For the Python wheel, supply absolute paths for all three `SCRFD_LOOM_*` asset
locations. Constructor arguments `weights=`, `kernels=` and `library=` take
precedence over environment variables. The CLI instead takes `--weights` and
`--kernels`, defaulting to `build/weights` and `build/kernels` relative to its
working directory.

## Python distribution

```bash
python -m pip install build
python -m build
```

This creates the loader wheel and source distribution in `dist/`. Neither
contains the native source tree or device assets; use a repository checkout to
build those. Both distributions include the project license and third-party
license notices. The build backend requires setuptools 77.0.3 or newer.
