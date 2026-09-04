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
from export_weights import pack

CV = "scrfd.conv3x3_f16_wmma"


def run_case(tmp, rng, name, batch, cin, cout, h, w, stride, weight=None, bias=None):
    cout_real = cout
    if weight is None:
        weight = (rng.standard_normal((cout, cin, 3, 3)) * (1.0 / np.sqrt(9 * cin))).astype(np.float32)
        bias = (rng.standard_normal(cout) * 0.1).astype(np.float32)
    w16, b32, info = pack(weight, bias)
    cp, k_pad, cout_pad = info["cin_pad"], info["k_pad"], info["cout_pad"]
    ho, wo = h // stride, w // stride
    m = batch * ho * wo

    x = (rng.standard_normal((batch, cin, h, w)) * 0.5).astype(np.float32)
    x_nhwc = np.zeros((batch, h, w, cp), np.float16)
    x_nhwc[..., :cin] = x.transpose(0, 2, 3, 1).astype(np.float16)

    hsaco = tmp / f"conv_{name}.hsaco"
    compile_kernel(ROOT / "kernels/conv3x3_f16_wmma.loom", "scrfd_conv3x3_f16_wmma",
                   {f"{CV}.height": h, f"{CV}.width": w, f"{CV}.stride": stride,
                    f"{CV}.cin_pad": cp, f"{CV}.k_size": k_pad, f"{CV}.n_size": cout_pad}, hsaco)
    (y,), t = launch(hsaco, "scrfd_conv3x3_f16_wmma", (cout_pad // 64, (m + 63) // 64, 1), (256, 1, 1),
                     [("i32", m), ("in_f16", x_nhwc.reshape(-1, cp)), ("in_f16", w16), ("in", b32),
                      ("out_f16", ((m, cout_pad), np.float16))], tmp, repeat=10)

    want = R.conv2d(x_nhwc[..., :cin].astype(np.float64).transpose(0, 3, 1, 2), weight, bias, stride, 1)
    want = want.transpose(0, 2, 3, 1).reshape(m, cout_real)
    flops_real = 2.0 * m * 9 * cin * cout_real
    us = t["per_launch_us"]
    return report(f"{name:<10s} {cin:>3d}->{cout_real:<3d} {h:>3d}x{w:<3d} s{stride} B={batch}  "
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
        for name, batch in (("c00", 2), ("c03", 4), ("c10", 4), ("c11", 4), ("c27", 4)):
            op = convs[name]
            _, cin, h, w = graph.shapes[op.inputs[0]]
            ok &= run_case(tmp, rng, name, batch, cin, op.cout, h, w, op.stride, op.weight, op.bias)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
