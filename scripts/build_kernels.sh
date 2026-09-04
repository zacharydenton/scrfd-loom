#!/usr/bin/env bash
# Compile every kernel the detector launches, at the configuration it launches it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
final=build/kernels
mkdir -p build "$final"
# Build a complete set off to the side. Only after every compile succeeds do we
# replace the HSACOs in the public directory, so removed configurations cannot
# linger and a failed build leaves the previous complete set intact.
out=$(mktemp -d build/.kernels.XXXXXX)
trap 'rm -rf "$out"' EXIT

compile() { # source-stem root-symbol output-stem config...
  local src="kernels/$1.loom" root="$2" stem="$3"; shift 3
  local args=()
  for c in "$@"; do args+=("--config=$c"); done
  "$LOOM_COMPILE" "$src" --backend=amdgpu-hal --target="$LOOM_TARGET" \
    --root="@$root" "${args[@]}" --output="$out/$stem.hsaco"
  printf '  %-28s %s\n' "$stem" "$(stat -c%s "$out/$stem.hsaco") bytes"
}

echo "compiling for $LOOM_TARGET"

# Stem input: letterboxed BGR uint8 -> normalised RGB NHWC f16, 3 channels padded to 4.
compile hwc_u8_to_nhwc_f16 scrfd_hwc_u8_to_nhwc_f16 hwc_u8_to_nhwc scrfd.hwc_u8_to_nhwc_f16.size=640

# Everything the graph launches, one line per distinct (kernel, config), generated
# by tools/gen_launch_table.py from the ONNX file.
source scripts/kernels.generated

find "$final" -maxdepth 1 -type f -name '*.hsaco' -delete
mv "$out"/*.hsaco "$final"/
rmdir "$out"
trap - EXIT
