#!/usr/bin/env bash
# Compile every kernel the detector launches, at the configuration it launches it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
out=build/kernels
mkdir -p "$out"

compile() { # source-stem root-symbol output-stem config...
  local src="kernels/$1.loom" root="$2" stem="$3"; shift 3
  local args=()
  for c in "$@"; do args+=("--config=$c"); done
  "$LOOM_COMPILE" "$src" --backend=amdgpu-hal --target="$LOOM_TARGET" \
    --root="@$root" "${args[@]}" --output="$out/$stem.hsaco"
  printf '  %-28s %s\n' "$stem" "$(stat -c%s "$out/$stem.hsaco") bytes"
}

echo "compiling for $LOOM_TARGET"

# Stem input: NCHW f32 blob -> NHWC f16, 3 channels padded to 4.
compile nchw_to_nhwc_f16 scrfd_nchw_to_nhwc_f16 nchw_to_nhwc \
  scrfd.nchw_to_nhwc_f16.size=640 scrfd.nchw_to_nhwc_f16.channels=3 \
  scrfd.nchw_to_nhwc_f16.channels_pad=4
