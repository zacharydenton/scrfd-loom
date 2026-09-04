"""pool2_f16 (max and mean, 2x2 stride 2) vs NumPy, at a tiny shape and at the
real stem shape, with a batch."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

P = "scrfd.pool2_f16"


def main() -> int:
    ok = True
    rng = np.random.default_rng(17)
    with workdir() as tmp:
        tmp = Path(tmp)
        for name, batch, h, w, c in (("tiny", 2, 4, 6, 8), ("stem 320^2", 2, 320, 320, 56), ("80^2", 2, 80, 80, 88)):
            for take_max in (1, 0):
                hsaco = tmp / f"pool_{h}_{c}_{take_max}.hsaco"
                compile_kernel(ROOT / "kernels/pool2_f16.loom", "scrfd_pool2_f16",
                               {f"{P}.height": h, f"{P}.width": w, f"{P}.channels": c,
                                f"{P}.take_max": take_max}, hsaco)
                x = rng.standard_normal((batch, h, w, c)).astype(np.float16)
                ho, wo = h // 2, w // 2
                m = batch * ho * wo
                (y,), t = launch(hsaco, "scrfd_pool2_f16", (batch * ho, 1, 1), (256, 1, 1),
                                 [("i32", m), ("in_f16", x.reshape(-1, c)),
                                  ("out_f16", ((m, c), np.float16))], tmp, repeat=10)
                v = x.astype(np.float64).reshape(batch, ho, 2, wo, 2, c)
                want = (v.max(axis=(2, 4)) if take_max else v.mean(axis=(2, 4))).reshape(m, c)
                kind = "max " if take_max else "mean"
                ok &= report(f"{kind} {name:<11s} B={batch} {h}x{w}x{c} ({t['per_launch_us']:7.2f} us)",
                             y, want, atol=1e-3 if take_max else 2e-3, rtol=1e-3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
