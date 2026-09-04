"""nchw_to_nhwc_f16 vs NumPy: exact relayout, channel padding to 4, batch of 2."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

P = "scrfd.nchw_to_nhwc_f16"


def main() -> int:
    ok = True
    rng = np.random.default_rng(3)
    with workdir() as tmp:
        tmp = Path(tmp)
        for size, batch in ((16, 2), (640, 2)):
            hsaco = tmp / f"cvt_{size}.hsaco"
            compile_kernel(ROOT / "kernels/nchw_to_nhwc_f16.loom", "scrfd_nchw_to_nhwc_f16",
                           {f"{P}.size": size, f"{P}.channels": 3, f"{P}.channels_pad": 4}, hsaco)
            x = rng.standard_normal((batch, 3, size, size)).astype(np.float32)
            rows = batch * size
            (out,), timing = launch(hsaco, "scrfd_nchw_to_nhwc_f16", (rows, 1, 1), (256, 1, 1),
                                    [("i32", rows), ("in", x),
                                     ("out_f16", ((batch * size * size, 4), np.float16))],
                                    tmp, repeat=20)
            want = np.zeros((batch, size, size, 4), np.float64)
            want[..., :3] = x.transpose(0, 2, 3, 1).astype(np.float16)
            ok &= report(f"size={size} batch={batch} ({timing['per_launch_us']:7.2f} us, "
                         f"{batch * 3 * size * size * 4 / (timing['per_launch_us'] * 1e-6) / 1e9:5.1f} GB/s in)",
                         out, want.reshape(-1, 4), atol=0, rtol=0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
