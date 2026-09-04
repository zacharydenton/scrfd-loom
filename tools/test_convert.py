"""hwc_u8_to_nhwc_f16 vs NumPy: the letterboxed BGR uint8 image -> normalised RGB
NHWC f16 padded to 4 channels, bit-exact against blobFromImage's arithmetic."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

P = "scrfd.hwc_u8_to_nhwc_f16"
SYMBOL = "scrfd_hwc_u8_to_nhwc_f16"


def expected(image: np.ndarray) -> np.ndarray:
    """(x - 127.5) / 128 in f32 with swapRB, truncated to f16, as blobFromImage + the old
    f32 converter produced it; the fourth channel is zero."""
    batch, size = image.shape[0], image.shape[1]
    rgb = image[..., ::-1].astype(np.float32)
    out = np.zeros((batch, size, size, 4), np.float16)
    out[..., :3] = ((rgb - np.float32(127.5)) * np.float32(1 / 128)).astype(np.float16)
    return out


def main() -> int:
    ok = True
    rng = np.random.default_rng(3)
    with workdir() as tmp:
        tmp = Path(tmp)
        for size, batch in ((16, 2), (640, 2)):
            hsaco = tmp / f"cvt_{size}.hsaco"
            compile_kernel(ROOT / "kernels/hwc_u8_to_nhwc_f16.loom", SYMBOL, {f"{P}.size": size}, hsaco)
            image = rng.integers(0, 256, (batch, size, size, 3), dtype=np.uint8)
            # the extremes must survive too
            image[0, 0, 0] = (0, 255, 128)
            rows = batch * size
            (out,), timing = launch(hsaco, SYMBOL, (rows, 1, 1), (256, 1, 1),
                                    [("i32", rows), ("in_u8", image),
                                     ("out_f16", ((batch * size * size, 4), np.float16))],
                                    tmp, repeat=20)
            want = expected(image).reshape(-1, 4)
            ok &= report(f"size={size} batch={batch} ({timing['per_launch_us']:7.2f} us, "
                         f"{batch * size * size * 3 / (timing['per_launch_us'] * 1e-6) / 1e9:5.1f} GB/s in)",
                         out, want, atol=0, rtol=0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
