#!/usr/bin/env bash
# The one test command. Formats, builds every kernel, checks the generated files
# against their generators, rebuilds the host, runs every unit test against a
# float64 reference on the real GPU, checks the runner's error paths, then
# validates the whole detector against onnxruntime and insightface's detections.
#
#   scripts/test.sh          everything
#   scripts/test.sh --quick  skip the onnxruntime/insightface comparisons
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

quick=0
[ "${1:-}" = "--quick" ] && quick=1
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
export tmpdir

status=0
step() {
  local name="$1"; shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then printf '  ok\n'; else printf '  FAILED: %s\n' "$name"; status=1; fi
}

step "loom sources are canonically formatted" bash -c '"$LOOM_FORMAT" --check kernels/*.loom'
step "generated kernels match their generator" bash -c '
  python3 tools/gen_variants.py --output-dir "$tmpdir" >/dev/null &&
  for f in conv3x3_f16_wmma_relu conv3x3_f16_wmma_add conv3x3_f16_wmma_relu_add matmul_add_resized_f16_wmma; do
    "$LOOM_FORMAT" --in-place "$tmpdir/$f.loom" >/dev/null && cmp -s "$tmpdir/$f.loom" "kernels/$f.loom" || { echo "  $f differs"; exit 1; }
  done'
step "launch table matches the graph" bash -c '
  cp host/graph_table.inc "$tmpdir/table.inc" && cp scripts/kernels.generated "$tmpdir/kernels.generated" &&
  python3 tools/gen_launch_table.py >/dev/null &&
  cmp -s host/graph_table.inc "$tmpdir/table.inc" && cmp -s scripts/kernels.generated "$tmpdir/kernels.generated"'
step "export weights" python3 tools/export_weights.py
step "build kernels" ./scripts/build_kernels.sh
step "build host programs" bash -c '
  /opt/rocm/bin/hipcc -O2 -Wall -Werror -o host/loomrun host/loomrun.cpp &&
  /opt/rocm/bin/hipcc -O2 -Wall -Werror -o host/scrfd host/scrfd.cpp &&
  /opt/rocm/bin/hipcc -O2 -Wall -Werror -shared -fPIC -DSCRFD_NO_MAIN -o host/libscrfd.so host/scrfd.cpp'

step "nchw -> nhwc"          python3 tools/test_nchw_to_nhwc.py
step "im2col"                python3 tools/test_im2col.py
step "pools"                 python3 tools/test_pool2.py
step "conv3x3 + variants"    python3 tools/test_conv3x3.py
step "fpn lateral"           python3 tools/test_fpn_lateral.py
step "explicit conv (fallback)" python3 tools/test_conv_explicit.py

step "runner rejects bad input" bash -c '
  rejects() {   # <phrase> <command...>: must exit 64 AND print the phrase
    local want="$1"; shift
    local out; out=$("$@" 2>&1); local rc=$?
    if [ "$rc" != 64 ]; then echo "  expected exit 64, got $rc: $*"; return 1; fi
    if ! grep -q -- "$want" <<<"$out"; then echo "  missing \"$want\" in: $out"; return 1; fi
  }
  printf short > "$tmpdir/short.bin"
  python3 -c "import numpy as np; np.zeros(2*3*640*640, np.float32).tofile(\"$tmpdir/two.bin\")"
  rejects "needs a value"     ./host/scrfd --batch                                   &&
  rejects "must be 1"         ./host/scrfd --input "$tmpdir/two.bin" --batch 0       &&
  rejects "input is required" ./host/scrfd                                           &&
  rejects "not a multiple"    ./host/scrfd --input "$tmpdir/short.bin"               &&
  rejects "holds 2 images"    ./host/scrfd --input "$tmpdir/two.bin" --batch 3'
step "session rejects a lying manifest" bash -c '
  cp -r build/weights "$tmpdir/badw" && sed -i "s/^c03 \([0-9]*\) [0-9]*$/c03 \1 1/" "$tmpdir/badw/manifest_f16.txt" &&
  python3 -c "import numpy as np; np.zeros(3*640*640, np.float32).tofile(\"$tmpdir/one.bin\")" &&
  out=$(./host/scrfd --weights "$tmpdir/badw" --input "$tmpdir/one.bin" 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -q "the kernel expects" <<<"$out"'

if [ "$quick" = 0 ]; then
  # torch is not involved here, but onnxruntime's MIGraphX build loads its own
  # ROCm; keep the Loom runtime path off these steps as dinov3-loom does.
  step "reference vs onnxruntime, decode vs insightface" env -u LD_LIBRARY_PATH python3 tools/test_reference.py
  step "end to end vs onnxruntime and insightface"        env -u LD_LIBRARY_PATH python3 tools/validate.py
  step "python api, batched, distinct images"             env -u LD_LIBRARY_PATH python3 tools/test_api.py
fi

printf '\n'
[ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
