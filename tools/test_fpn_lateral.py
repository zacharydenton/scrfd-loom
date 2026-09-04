"""matmul_add_resized: the FPN lateral 1x1 conv with the nearest-2x-upsampled
coarser level added in the epilogue, vs NumPy. Exercises both real laterals:
88->56 at 80^2 (input stored 128 wide) and 88->56 at 40^2."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
from export_weights import pack, storage_stride

P = "scrfd.matmul_add_resized_f16_wmma"


def main() -> int:
    ok = True
    rng = np.random.default_rng(31)
    with workdir() as tmp:
        tmp = Path(tmp)
        # cin >= 9: a narrower tensor is stored 8 wide (only the stem image is), and a
        # 1x1 conv's K must be the row stride, a multiple of 32.
        for name, batch, cin, cout, h, w in (("tiny", 2, 16, 64, 4, 6), ("80^2 lateral", 2, 88, 56, 80, 80),
                                            ("40^2 lateral", 2, 88, 56, 40, 40)):
            weight = (rng.standard_normal((cout, cin, 1, 1)) / np.sqrt(cin)).astype(np.float32)
            bias = (rng.standard_normal(cout) * 0.1).astype(np.float32)
            w16, b32, info = pack(weight, bias)
            k_pad, cout_pad = info["k_pad"], info["cout_pad"]
            cs = storage_stride(cin)
            assert k_pad == cs, (k_pad, cs)          # a 1x1 conv's K is the input's row stride
            m = batch * h * w
            x = np.zeros((m, cs), np.float16)
            x[:, :cin] = (rng.standard_normal((m, cin)) * 0.5).astype(np.float16)
            coarse = rng.standard_normal((batch * (h // 2) * (w // 2), cout_pad)).astype(np.float16)

            hsaco = tmp / f"lat_{name}.hsaco"
            compile_kernel(ROOT / "kernels/matmul_add_resized_f16_wmma.loom", "scrfd_matmul_add_resized_f16_wmma",
                           {f"{P}.k_size": k_pad, f"{P}.n_size": cout_pad, f"{P}.height": h, f"{P}.width": w}, hsaco)
            (y,), t = launch(hsaco, "scrfd_matmul_add_resized_f16_wmma", (cout_pad // 64, (m + 63) // 64, 1),
                             (256, 1, 1), [("i32", m), ("in_f16", x), ("in_f16", w16), ("in", b32),
                                           ("out_f16", ((m, cout_pad), np.float16)), ("in_f16", coarse)],
                             tmp, repeat=10)
            fine = x[:, :cin].astype(np.float64) @ weight.reshape(cout, cin).astype(np.float64).T + bias
            up = coarse.astype(np.float64).reshape(batch, h // 2, w // 2, cout_pad)
            up = up.repeat(2, axis=1).repeat(2, axis=2).reshape(m, cout_pad)[:, :cout]
            ok &= report(f"{name:<13s} B={batch} {cin}->{cout} {h}x{w} ({t['per_launch_us']:7.1f} us)",
                         y[:, :cout], fine + up, atol=5e-2, rtol=5e-2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
