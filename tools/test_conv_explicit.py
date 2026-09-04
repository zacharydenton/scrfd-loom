"""Day-one convolution: im2col_f16 + the WMMA matmul, vs the float64 reference.

This chains the two kernels exactly as the runner will for the fallback path,
on real layer shapes with the real exported weights, and grades the result
against reference.conv2d in NCHW -- so a mistake in the gather order or the
weight layout shows up here, not in the model.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import graph as G
import reference as R
from export_weights import pack

IM = "scrfd.im2col_f16"
MM = "dinov3.matmul_bias_f16_wmma_af16_cf16"


def main() -> int:
    ok = True
    rng = np.random.default_rng(9)
    graph = G.load()
    convs = {op.name: op for op in graph.convs}
    # (conv, batch): the stem's stride-2 first conv, a 160^2 residual conv, the
    # first 88-wide stride-2 conv, and a 224-wide 20^2 conv.
    cases = [("c00", 1), ("c03", 2), ("c10", 2), ("c27", 2)]
    with workdir() as tmp:
        tmp = Path(tmp)
        for name, batch in cases:
            op = convs[name]
            w16, b32, info = pack(op.weight, op.bias)
            cp, k_pad, cout_pad = info["cin_pad"], info["k_pad"], info["cout_pad"]
            _, cin, h, w = graph.shapes[op.inputs[0]]
            s = op.stride
            ho, wo = h // s, w // s
            m = batch * ho * wo

            x = (rng.standard_normal((batch, cin, h, w)) * 0.5).astype(np.float32)
            x_nhwc = np.zeros((batch, h, w, cp), np.float16)
            x_nhwc[..., :cin] = x.transpose(0, 2, 3, 1).astype(np.float16)

            h_im = tmp / f"im2col_{name}.hsaco"
            compile_kernel(ROOT / "kernels/im2col_f16.loom", "scrfd_im2col_f16",
                           {f"{IM}.height": h, f"{IM}.width": w, f"{IM}.stride": s,
                            f"{IM}.cin_pad": cp, f"{IM}.k_pad": k_pad}, h_im)
            (cols,), t_im = launch(h_im, "scrfd_im2col_f16", (batch * ho, 1, 1), (256, 1, 1),
                                   [("i32", m), ("in_f16", x_nhwc.reshape(-1, cp)),
                                    ("out_f16", ((m, k_pad), np.float16))], tmp, repeat=5)

            h_mm = tmp / f"mm_{name}.hsaco"
            compile_kernel(ROOT / "kernels/matmul_bias_f16_wmma_af16_cf16.loom",
                           "dinov3_matmul_bias_f16_wmma_af16_cf16",
                           {f"{MM}.k_size": k_pad, f"{MM}.n_size": cout_pad}, h_mm)
            (y,), t_mm = launch(h_mm, "dinov3_matmul_bias_f16_wmma_af16_cf16",
                                (cout_pad // 64, (m + 63) // 64, 1), (256, 1, 1),
                                [("i32", m), ("in_f16", cols), ("in_f16", w16), ("in", b32),
                                 ("out_f16", ((m, cout_pad), np.float16))], tmp, repeat=5)

            want = R.conv2d(x_nhwc[..., :cin].astype(np.float64).transpose(0, 3, 1, 2),
                            op.weight, op.bias, s, op.pad)            # [B,Cout,Ho,Wo]
            want = want.transpose(0, 2, 3, 1).reshape(m, op.cout)
            got = y[:, :op.cout]
            flops = 2.0 * m * k_pad * cout_pad
            ok &= report(f"{name} {cin:>3d}->{op.cout:<3d} {h}x{w} s{s} B={batch}  "
                         f"(im2col {t_im['per_launch_us']:7.1f} us + matmul {t_mm['per_launch_us']:7.1f} us, "
                         f"{flops / (t_mm['per_launch_us'] * 1e-6) / 1e12:5.1f} TFLOP/s)",
                         got, want, atol=5e-2, rtol=5e-2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
