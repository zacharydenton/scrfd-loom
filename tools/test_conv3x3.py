"""conv3x3_f16_wmma (implicit GEMM) vs the float64 reference, on real layers
with the real exported weights, and against the explicit im2col+matmul path's
timing so the gather's cost is visible from day one.

A tiny 4x6 case with Cin 4 exercises every halo position; the real shapes cover
stride 1 and 2, the stem, a 56-wide 160^2 layer, the 88-wide 80^2 workhorse
(7 layers, 23% of all FLOPs) and the 224-wide 20^2 layers.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import graph as G
import reference as R
from export_weights import pack, storage_stride

CV = "scrfd.conv3x3_f16_wmma"


def run_case(tmp, rng, name, batch, cin, cout, h, w, stride, weight=None, bias=None, variant="", tile=64):
    cout_real = cout
    if weight is None:
        weight = (rng.standard_normal((cout, cin, 3, 3)) * (1.0 / np.sqrt(9 * cin))).astype(np.float32)
        bias = (rng.standard_normal(cout) * 0.1).astype(np.float32)
    w16, b32, info = pack(weight, bias, cout_align=tile)
    cp, k_pad, cout_pad = info["cin_pad"], info["k_pad"], info["cout_pad"]
    cs = storage_stride(cin)          # the input is stored at its producer's width
    ho, wo = h // stride, w // stride
    m = batch * ho * wo

    x = (rng.standard_normal((batch, cin, h, w)) * 0.5).astype(np.float32)
    x_nhwc = np.zeros((batch, h, w, cs), np.float16)
    x_nhwc[..., :cin] = x.transpose(0, 2, 3, 1).astype(np.float16)

    suffix = f"_{variant}" if variant else ""
    base = "conv3x3_n128_f16_wmma" if tile == 128 else "conv3x3_f16_wmma"
    ns, symbol = f"scrfd.{base}{suffix}", f"scrfd_{base}{suffix}"
    hsaco = tmp / f"{base}{suffix}_{name}.hsaco"
    compile_kernel(ROOT / f"kernels/{base}{suffix}.loom", symbol,
                   {f"{ns}.height": h, f"{ns}.width": w, f"{ns}.stride": stride,
                    f"{ns}.cin_pad": cp, f"{ns}.cin_stride": cs,
                    f"{ns}.k_size": k_pad, f"{ns}.n_size": cout_pad}, hsaco)
    args = [("i32", m), ("in_f16", x_nhwc.reshape(-1, cs)), ("in_f16", w16), ("in", b32),
            ("out_f16", ((m, cout_pad), np.float16))]
    residual = None
    if "add" in variant:
        residual = rng.standard_normal((m, cout_pad)).astype(np.float16)
        args.append(("in_f16", residual))
    (y,), t = launch(hsaco, symbol, (cout_pad // tile, (m + 63) // 64, 1), (256, 1, 1), args, tmp, repeat=10)

    want = R.conv2d(x_nhwc[..., :cin].astype(np.float64).transpose(0, 3, 1, 2), weight, bias, stride, 1)
    want = want.transpose(0, 2, 3, 1).reshape(m, cout_real)
    if residual is not None:
        want = want + residual[:, :cout_real].astype(np.float64)
    if "relu" in variant:
        want = np.maximum(want, 0.0)
    flops_real = 2.0 * m * 9 * cin * cout_real
    us = t["per_launch_us"]
    return report(f"{name:<8s} {'n128 ' if tile == 128 else ''}{variant or 'plain':<8s} {cin:>3d}->{cout_real:<3d} {h:>3d}x{w:<3d} s{stride} B={batch}  "
                  f"({us:8.1f} us, {flops_real / (us * 1e-6) / 1e12:5.1f} TFLOP/s useful)",
                  y[:, :cout_real], want, atol=5e-2, rtol=5e-2)


def main() -> int:
    ok = True
    rng = np.random.default_rng(21)
    graph = G.load()
    convs = {op.name: op for op in graph.convs}
    with workdir() as tmp:
        tmp = Path(tmp)
        ok &= run_case(tmp, rng, "tiny s1", 2, 4, 64, 4, 6, 1)
        ok &= run_case(tmp, rng, "tiny s2", 2, 4, 64, 4, 6, 2)
        # Cin=1024 makes K=9216 and exercises the upper half of the declared
        # k_size range, which must agree with the optimizer's index assumption.
        ok &= run_case(tmp, rng, "wide k", 1, 1024, 64, 2, 2, 1)
        # Every epilogue variant, on the tiny shape (all halo cases) and one real layer.
        for variant in ("relu", "add", "relu_add"):
            ok &= run_case(tmp, rng, "tiny s1", 2, 4, 64, 4, 6, 1, variant=variant)
        # The 64x128 tile family, plain and with the fused residual, on the tiny shape.
        ok &= run_case(tmp, rng, "tiny s1", 2, 8, 128, 4, 6, 1, tile=128)
        ok &= run_case(tmp, rng, "tiny s2", 2, 8, 128, 4, 6, 2, variant="relu_add", tile=128)
        for name, batch in (("c00", 2), ("c03", 4), ("c10", 4), ("c18", 4), ("c27", 4)):
            op = convs[name]
            assert op.ksize == 3, f"{name} is a {op.ksize}x{op.ksize} conv"
            _, cin, h, w = graph.shapes[op.inputs[0]]
            variant = "relu_add" if name == "c03" else ""
            ok &= run_case(tmp, rng, name, batch, cin, op.cout, h, w, op.stride, op.weight, op.bias, variant)
            if name in ("c10", "c18"):   # the 88-channel layers run on the 64x128 tile
                ok &= run_case(tmp, rng, name, batch, cin, op.cout, h, w, op.stride, op.weight, op.bias, "relu_add" if name == "c10" else "relu", tile=128)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
