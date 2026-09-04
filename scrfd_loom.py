"""Drop-in replacement for insightface's SCRFD `det_10g` detector.

    from scrfd_loom import SCRFDLoom

    model = SCRFDLoom()
    det, kps = model.detect(image_bgr)            # exactly what SCRFD.detect returns
    results = model.detect_batch(list_of_images)  # [(det, kps), ...], one GPU call per max_batch
    model.close()

``det`` is ``(n, 5)`` ``[x1, y1, x2, y2, score]`` in original-image pixels and
``kps`` is ``(n, 5, 2)``, after NMS, as ``SCRFD.detect`` returns them. Results
are score-sorted unless ``max_num`` applies InsightFace's final area/centre
ranking. The letterbox is insightface's; the normalisation, the network and the
anchor decode run in the native session; the runtime NMS is packaged in
``scrfd_loom_decode.py``.

Importing this module does not initialize HIP. Each model owns an independent
native session; construction loads the kernels and weights, and every call
reuses its GPU allocations. Calls on one object are serialized and may come
from multiple threads.
"""
from __future__ import annotations

import ctypes
import math
import operator
import os
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
import scrfd_loom_decode as D

SIZE = 640
CANDIDATE_FLOATS = 15
MAX_BATCH = 64
DEFAULT_MAX_BATCH = 16
DEFAULT_MAX_CANDIDATES = 4096
DET_THRESH, NMS_THRESH = D.DET_THRESH, D.NMS_THRESH

_ABI_VERSION = 2
_ERROR_CAPACITY = 4096
_U8Pointer = ctypes.POINTER(ctypes.c_uint8)
_FloatPointer = ctypes.POINTER(ctypes.c_float)
_I32Pointer = ctypes.POINTER(ctypes.c_int32)


class SCRFDError(RuntimeError):
    """The native SCRFD runtime could not initialize or complete a call."""


def letterbox_into(image_bgr: np.ndarray, canvas: np.ndarray) -> float:
    """insightface's letterbox (tools/decode.py) written into a preallocated
    (SIZE, SIZE, 3) uint8 canvas: resize to fit, top-left aligned, the rest black.
    Returns det_scale."""
    import cv2
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 BGR image, got {image_bgr.shape} {image_bgr.dtype}")
    if image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0:
        raise ValueError("image height and width must be positive")
    if (canvas.ndim != 3 or canvas.shape[0] == 0 or canvas.shape[0] != canvas.shape[1]
            or canvas.shape[2] != 3 or canvas.dtype != np.uint8):
        raise ValueError(f"expected a square (size, size, 3) uint8 BGR canvas, got {canvas.shape} {canvas.dtype}")
    size = canvas.shape[0]
    im_ratio = float(image_bgr.shape[0]) / image_bgr.shape[1]
    if im_ratio > 1.0:
        new_height = size
        new_width = int(new_height / im_ratio)
    else:
        new_width = size
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / image_bgr.shape[0]
    canvas[:new_height, :new_width] = cv2.resize(image_bgr, (new_width, new_height))
    canvas[new_height:] = 0
    canvas[:new_height, new_width:] = 0
    return det_scale


def _path(value: str | os.PathLike[str] | None, environment: str, default: Path) -> Path:
    if value is None:
        value = os.environ.get(environment, default)
    return Path(value).expanduser().resolve()


def _message(error: ctypes.Array[ctypes.c_char], fallback: str) -> str:
    return error.value.decode("utf-8", errors="replace") or fallback


class SCRFDLoom:
    """A resident, reusable Loom detection session.

    Calls on one object are serialized and may safely come from multiple Python
    threads. Use separate objects for independent sessions. A session inherited
    across ``fork()`` is deliberately rejected because HIP state is not fork-safe.
    """

    def __init__(
        self,
        weights: str | os.PathLike[str] | None = None,
        kernels: str | os.PathLike[str] | None = None,
        library: str | os.PathLike[str] | None = None,
        max_batch: int = DEFAULT_MAX_BATCH,
        det_thresh: float = DET_THRESH,
        nms_thresh: float = NMS_THRESH,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        try:
            max_batch = operator.index(max_batch)
            max_candidates = operator.index(max_candidates)
        except TypeError as failure:
            raise TypeError("max_batch and max_candidates must be integers") from failure
        if not 1 <= max_batch <= MAX_BATCH:
            raise ValueError(f"max_batch must be in 1..{MAX_BATCH}, got {max_batch}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be positive, got {max_candidates}")
        try:
            det_thresh, nms_thresh = float(det_thresh), float(nms_thresh)
        except (TypeError, ValueError) as failure:
            raise TypeError("det_thresh and nms_thresh must be real numbers") from failure
        if not math.isfinite(det_thresh) or not 0.0 < det_thresh < 1.0:
            raise ValueError(f"det_thresh must be strictly between 0 and 1, got {det_thresh}")
        if not math.isfinite(nms_thresh) or not 0.0 <= nms_thresh <= 1.0:
            raise ValueError(f"nms_thresh must be finite and between 0 and 1, got {nms_thresh}")
        self.det_thresh, self.nms_thresh = det_thresh, nms_thresh

        self.weights = _path(weights, "SCRFD_LOOM_WEIGHTS", ROOT / "build/weights")
        self.kernels = _path(kernels, "SCRFD_LOOM_KERNELS", ROOT / "build/kernels")
        self.library = _path(library, "SCRFD_LOOM_LIBRARY", ROOT / "build/libscrfd.so")
        for path, hint in (
            (self.library, "./scripts/build_host.sh"),
            (self.kernels, "./scripts/build_kernels.sh"),
            (self.weights, "python3 tools/export_weights.py"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; run: {hint}")

        try:
            native = ctypes.CDLL(self.library)
        except OSError as failure:
            raise SCRFDError(f"cannot load {self.library}: {failure}") from failure
        try:
            self._configure(native)
        except AttributeError as failure:
            raise SCRFDError(
                f"{self.library} does not expose the SCRFD ABI; rebuild it with ./scripts/build_host.sh"
            ) from failure
        abi = native.scrfd_abi_version()
        if abi != _ABI_VERSION:
            raise SCRFDError(
                f"{self.library} uses ABI {abi}; Python expects ABI {_ABI_VERSION}; "
                "rebuild it with ./scripts/build_host.sh"
            )
        self.size = native.scrfd_input_size()

        # The reusable host buffers come before the native session so a rare
        # NumPy allocation failure cannot strand an already-created GPU session.
        self._input = np.zeros((max_batch, self.size, self.size, 3), np.uint8)
        self._candidates = np.empty((max_batch, max_candidates, CANDIDATE_FLOATS), np.float32)
        self._counts = np.empty(max_batch, np.int32)
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._native = native
        self._handle: ctypes.c_void_p | None = None

        handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = native.scrfd_create(
            os.fsencode(self.weights), os.fsencode(self.kernels), max_batch,
            ctypes.byref(handle), error, len(error),
        )
        if status:
            raise SCRFDError(_message(error, f"scrfd_create failed with status {status}"))
        if not handle:
            raise SCRFDError("scrfd_create succeeded without returning a session")
        self._handle = handle
        self.max_batch = max_batch
        self.max_candidates = max_candidates

    @staticmethod
    def _configure(native: ctypes.CDLL) -> None:
        native.scrfd_abi_version.argtypes = []
        native.scrfd_abi_version.restype = ctypes.c_uint32
        native.scrfd_input_size.argtypes = []
        native.scrfd_input_size.restype = ctypes.c_int
        native.scrfd_create.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
        ]
        native.scrfd_create.restype = ctypes.c_int
        native.scrfd_run.argtypes = [
            ctypes.c_void_p,
            _U8Pointer, ctypes.c_size_t, ctypes.c_int, ctypes.c_float,
            _FloatPointer, ctypes.c_size_t,
            _I32Pointer, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
        ]
        native.scrfd_run.restype = ctypes.c_int
        native.scrfd_max_batch.argtypes = [ctypes.c_void_p]
        native.scrfd_max_batch.restype = ctypes.c_int
        native.scrfd_destroy.argtypes = [ctypes.c_void_p]
        native.scrfd_destroy.restype = None

    @property
    def closed(self) -> bool:
        """Whether this model has released its native session."""
        return self._handle is None

    def _ensure_usable(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise SCRFDError("this SCRFDLoom session is closed")
        if os.getpid() != self._pid:
            raise SCRFDError("this SCRFDLoom session was inherited across fork; create a new model in the child process")
        return self._handle

    def close(self) -> None:
        """Release all GPU allocations and loaded modules; safe to call twice."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            handle = getattr(self, "_handle", None)
            self._handle = None
            # Destroying inherited HIP state in a forked child is itself unsafe.
            if handle is not None and os.getpid() == self._pid:
                self._native.scrfd_destroy(handle)

    def __enter__(self) -> SCRFDLoom:
        self._ensure_usable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors run during interpreter teardown, when module globals
            # and the dynamic loader may already be partly dismantled.
            pass

    # --- insightface-compatible API ---------------------------------------------
    def _call_options(self, input_size, max_num) -> int:
        if input_size is not None:
            try:
                requested = tuple(operator.index(v) for v in input_size)
            except (TypeError, ValueError) as failure:
                raise ValueError("input_size must be None or a (width, height) pair") from failure
            if requested != (self.size, self.size):
                raise ValueError(f"this fixed-shape build only supports input_size=({self.size}, {self.size})")
        if isinstance(max_num, bool):
            raise TypeError("max_num must be an integer")
        try:
            max_num = operator.index(max_num)
        except TypeError as failure:
            raise TypeError("max_num must be an integer") from failure
        if max_num < 0:
            raise ValueError("max_num must be non-negative")
        return max_num

    def detect_batch(self, images_bgr, max_num: int = 0, metric: str = "default",
                     input_size=None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Many images, ``max_batch`` per GPU call. Returns ``[(det, kps), ...]``."""
        max_num = self._call_options(input_size, max_num)
        images = list(images_bgr)
        with self._lock:
            self._ensure_usable()
            results = []
            for start in range(0, len(images), self.max_batch):
                results.extend(self._run(images[start:start + self.max_batch], max_num, metric))
        return results

    def detect(self, image_bgr: np.ndarray, input_size=None, max_num: int = 0,
               metric: str = "default") -> tuple[np.ndarray, np.ndarray]:
        """InsightFace-compatible ``SCRFD.detect`` for this fixed 640x640 build."""
        return self.detect_batch([image_bgr], max_num=max_num, metric=metric, input_size=input_size)[0]

    def detect_letterboxed(self, canvases, det_scales=None, max_num: int = 0,
                           metric: str = "default", image_shapes=None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Images the caller has already letterboxed to ``(size, size, 3)`` uint8 BGR,
        top-left aligned, with whatever resampler it prefers -- a list of them, or one
        stacked ``(B, size, size, 3)`` array, which is used in place without a copy.

        ``det_scales`` is one ``new_height / original_height`` per image, the value
        insightface's letterbox returns, and maps boxes and keypoints back to
        original-image pixels; ``None`` returns them in canvas pixels. For the
        default ``max_num`` metric, pass corresponding ``(height, width)`` entries
        in ``image_shapes`` when ``det_scales`` is not ``None``. Use this when
        the rest of a pipeline already resamples its own way: the resampler moves
        detections a little (a bilinear-vs-lanczos difference is roughly 0.97 IoU and
        two pixels on a landmark), and matching what produced your existing
        embeddings matters more than matching insightface.
        """
        stacked = np.asarray(canvases)
        expected = (self.size, self.size, 3)
        if stacked.ndim != 4 or stacked.shape[1:] != expected or stacked.dtype != np.uint8:
            raise ValueError(f"expected (B, {self.size}, {self.size}, 3) uint8 BGR canvases, got {stacked.shape} {stacked.dtype}")
        batch = stacked.shape[0]
        scales = [1.0] * batch if det_scales is None else [float(v) for v in det_scales]
        if len(scales) != batch:
            raise ValueError(f"det_scales has {len(scales)} entries for {batch} canvases")
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
            raise ValueError("det_scales entries must be positive finite numbers")
        max_num = self._call_options(None, max_num)
        if image_shapes is None:
            shapes = [(self.size, self.size)] * batch if det_scales is None else [None] * batch
        else:
            shapes = []
            for shape in image_shapes:
                try:
                    height, width = (operator.index(v) for v in shape)
                except (TypeError, ValueError) as failure:
                    raise ValueError("image_shapes entries must be (height, width) pairs") from failure
                if height <= 0 or width <= 0:
                    raise ValueError("image_shapes entries must have positive dimensions")
                shapes.append((height, width))
            if len(shapes) != batch:
                raise ValueError(f"image_shapes has {len(shapes)} entries for {batch} canvases")
        if max_num > 0 and metric != "max" and any(shape is None for shape in shapes):
            raise ValueError("image_shapes is required with det_scales and the default max_num metric")
        with self._lock:
            self._ensure_usable()
            results = []
            for start in range(0, batch, self.max_batch):
                chunk = np.ascontiguousarray(stacked[start:start + self.max_batch])
                results.extend(self._infer(chunk, scales[start:start + self.max_batch], max_num, metric,
                                           shapes[start:start + self.max_batch]))
        return results

    def _run(self, images: list[np.ndarray], max_num: int, metric: str) -> list[tuple[np.ndarray, np.ndarray]]:
        scales = [letterbox_into(image, self._input[i]) for i, image in enumerate(images)]
        shapes = [(image.shape[0], image.shape[1]) for image in images]
        return self._infer(self._input[:len(images)], scales, max_num, metric, shapes)

    def _infer(self, inputs: np.ndarray, scales: list[float], max_num: int, metric: str,
               image_shapes: list[tuple[int, int] | None]) -> list[tuple[np.ndarray, np.ndarray]]:
        """One native call on a contiguous (batch, size, size, 3) uint8 array."""
        handle = self._ensure_usable()
        batch = inputs.shape[0]
        assert inputs.flags.c_contiguous and 1 <= batch <= self.max_batch
        candidates, counts = self._candidates[:batch], self._counts[:batch]
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = self._native.scrfd_run(
            handle,
            inputs.ctypes.data_as(_U8Pointer), inputs.nbytes, batch, self.det_thresh,
            candidates.ctypes.data_as(_FloatPointer), candidates.size,
            counts.ctypes.data_as(_I32Pointer), counts.size,
            error, len(error),
        )
        if status:
            raise SCRFDError(_message(error, f"scrfd_run failed with status {status}"))
        results = []
        for b, (scale, image_shape) in enumerate(zip(scales, image_shapes)):
            # The native buffers are reused, so every result owns its data.
            rows = candidates[b, :counts[b]].copy()
            if rows.shape[0] == 0:
                results.append((np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)))
                continue
            # insightface: sort by score, back to original-image pixels, NMS, cap.
            order = rows[:, 4].argsort()[::-1]
            rows = rows[order]
            rows[:, :4] /= scale
            rows[:, 5:] /= scale
            keep = D.nms(rows[:, :5], self.nms_thresh)
            det, kps = rows[keep, :5], rows[keep, 5:].reshape(-1, 5, 2)
            det, kps = D.limit_detections(det, kps, max_num, metric, image_shape)
            results.append((np.ascontiguousarray(det), np.ascontiguousarray(kps)))
        return results


__all__ = [
    "CANDIDATE_FLOATS",
    "DEFAULT_MAX_BATCH",
    "DEFAULT_MAX_CANDIDATES",
    "MAX_BATCH",
    "SIZE",
    "SCRFDError",
    "SCRFDLoom",
    "letterbox_into",
]
