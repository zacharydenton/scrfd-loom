"""Drop-in replacement for insightface's SCRFD `det_10g` detector.

    from scrfd_loom import SCRFDLoom
    model = SCRFDLoom()
    det, kps = model.detect(image_bgr)          # exactly what SCRFD.detect returns

`det` is (n, 5) [x1, y1, x2, y2, score] in original-image pixels, `kps` is
(n, 5, 2). Preprocessing (letterbox, blobFromImage), anchor decode and NMS are
insightface's own code, vendored in tools/decode.py; only the network runs in
Loom. `detect_batch` runs many images in one GPU call, which the deployed ONNX
graph cannot do.

The session is resident: kernels and weights load once, buffers are sized for
`max_batch` up front, and each call is one ctypes entry into host/libscrfd.so.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
import decode as D                                     # noqa: E402

STRIDES = (8, 16, 32)
ABI_VERSION = 1


class SCRFDLoom:
    def __init__(self, max_batch: int = 8, weights: str | Path | None = None,
                 kernels: str | Path | None = None, library: str | Path | None = None,
                 det_thresh: float = D.DET_THRESH, nms_thresh: float = D.NMS_THRESH):
        self.weights = Path(weights or ROOT / "build/weights").resolve()
        self.kernels = Path(kernels or ROOT / "build/kernels").resolve()
        library = Path(library or ROOT / "host/libscrfd.so").resolve()
        for path, hint in ((library, "hipcc -O2 -shared -fPIC -DSCRFD_NO_MAIN -o host/libscrfd.so host/scrfd.cpp"),
                           (self.kernels, "./scripts/build_kernels.sh"),
                           (self.weights, "python3 tools/export_weights.py")):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; run: {hint}")
        self.det_thresh, self.nms_thresh = det_thresh, nms_thresh

        lib = ctypes.CDLL(str(library))
        lib.scrfd_abi_version.restype = ctypes.c_uint32
        if lib.scrfd_abi_version() != ABI_VERSION:
            raise RuntimeError(f"libscrfd ABI {lib.scrfd_abi_version()} != expected {ABI_VERSION}")
        lib.scrfd_input_size.restype = ctypes.c_int
        lib.scrfd_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
        lib.scrfd_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t,
                                  ctypes.c_char_p, ctypes.c_size_t]
        lib.scrfd_destroy.argtypes = [ctypes.c_void_p]
        self._lib = lib
        self.size = lib.scrfd_input_size()
        self.max_batch = max_batch
        handle = ctypes.c_void_p()
        err = ctypes.create_string_buffer(512)
        if lib.scrfd_create(str(self.weights).encode(), str(self.kernels).encode(), max_batch,
                            ctypes.byref(handle), err, len(err)):
            raise RuntimeError(err.value.decode())
        self._handle = handle

    def __del__(self):
        lib, handle = getattr(self, "_lib", None), getattr(self, "_handle", None)
        if lib is not None and handle:
            lib.scrfd_destroy(handle)

    # --- the network ------------------------------------------------------------
    def heads(self, blob: np.ndarray) -> list[np.ndarray]:
        """[B,3,S,S] f32 blob -> three [B*H*W][64] f16 head tensors (scores pre-sigmoid)."""
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        if blob.ndim != 4 or blob.shape[1:] != (3, self.size, self.size):
            raise ValueError(f"expected (B, 3, {self.size}, {self.size}), got {blob.shape}")
        batch = blob.shape[0]
        if not 1 <= batch <= self.max_batch:
            raise ValueError(f"batch {batch} outside 1..{self.max_batch} (max_batch at construction)")
        outs = [np.empty(batch * (self.size // s) ** 2 * 64, np.uint16) for s in STRIDES]
        err = ctypes.create_string_buffer(512)
        rc = self._lib.scrfd_run(self._handle, blob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), blob.size,
                                 batch, *sum(([o.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), o.size]
                                              for o in outs), []), err, len(err))
        if rc:
            raise RuntimeError(err.value.decode())
        return [o.view(np.float16) for o in outs]

    def raw_outputs(self, blob: np.ndarray) -> list[list[np.ndarray]]:
        """Per image, the nine outputs exactly as the ONNX graph emits them."""
        from validate import heads_to_outputs
        return heads_to_outputs(self.heads(blob), blob.shape[0], self.size)

    # --- insightface-compatible API ---------------------------------------------
    def detect_batch(self, images_bgr: list[np.ndarray], max_num: int = 0):
        """Many images in one GPU call. Returns [(det, kps), ...], one per image."""
        results = []
        for start in range(0, len(images_bgr), self.max_batch):
            chunk = images_bgr[start:start + self.max_batch]
            prepared = [D.letterbox(img, self.size) for img in chunk]
            blob = np.concatenate([D.blob(det_img) for det_img, _ in prepared])
            for outs, (_, scale) in zip(self.raw_outputs(blob), prepared):
                results.append(D.postprocess(outs, scale, self.size, self.det_thresh, self.nms_thresh, max_num))
        return results

    def detect(self, image_bgr: np.ndarray, max_num: int = 0):
        return self.detect_batch([image_bgr], max_num)[0]
