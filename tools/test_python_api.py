"""Lifecycle, error-boundary and correctness tests for the resident Python API.

Correctness is graded three ways: detect() against insightface's own detections
(the fixture), detect_batch() on four distinct images against onnxruntime on the
same letterboxed input, and the native decode against the vendored Python decode
run on the very same head tensors, where the two must agree to float rounding."""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from scrfd_loom import SCRFDError, SCRFDLoom, letterbox_into, CANDIDATE_FLOATS  # noqa: E402
from test_reference import test_image, iou, FIXTURE                             # noqa: E402
from validate import heads_to_outputs, run_loom                                  # noqa: E402
import decode as D                                                               # noqa: E402
import graph as G                                                                # noqa: E402


def require_raises(kind, phrase: str, call) -> None:
    try:
        call()
    except kind as failure:
        assert phrase in str(failure), str(failure)
    else:
        raise AssertionError(f"expected {kind.__name__} containing {phrase!r}")


def same_detections(got, want, box_tol=None, score_tol=1e-2, kps_tol=1.0) -> bool:
    (det, kps), (det_w, kps_w) = got, want
    if det.shape != det_w.shape or kps.shape != kps_w.shape:
        return False
    for a, b, ka, kb in zip(det, det_w, kps, kps_w):
        close = (np.abs(a[:4] - b[:4]).max() < box_tol) if box_tol is not None else iou(a, b) > 0.99
        if not (close and abs(a[4] - b[4]) < score_tol and np.abs(ka - kb).max() < kps_tol):
            return False
    return True


def main() -> int:
    import cv2
    import onnxruntime as ort
    ok = True

    def check(name: str, good: bool) -> None:
        nonlocal ok
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} {name}")

    require_raises(ValueError, "1..64", lambda: SCRFDLoom(max_batch=65))
    require_raises(ValueError, "det_thresh", lambda: SCRFDLoom(det_thresh=1.5))
    require_raises(ValueError, "uint8", lambda: letterbox_into(np.zeros((4, 4, 3), np.float32), np.zeros((640, 640, 3), np.uint8)))
    # Native initialization failures return through the ABI, never exit().
    with tempfile.TemporaryDirectory() as empty:
        require_raises(SCRFDError, "manifest", lambda: SCRFDLoom(weights=empty, max_batch=1))
    with tempfile.TemporaryDirectory() as empty:
        require_raises(SCRFDError, "hipModuleLoad", lambda: SCRFDLoom(kernels=empty, max_batch=1))
    check("construction errors: ranges, dtype, native failures through the ABI", True)

    img = cv2.imread(str(test_image()))
    fx = json.loads(FIXTURE.read_text())
    fixture = (np.array(fx["det"], np.float32), np.array(fx["kps"], np.float32))

    with SCRFDLoom(max_batch=2) as model:
        assert model._native.scrfd_max_batch(model._handle) == 2
        det, kps = model.detect(img)
        check(f"detect(): {len(det)} faces vs insightface's {len(fixture[0])}", same_detections((det, kps), fixture))
        assert det.dtype == np.float32 and kps.shape == (len(det), 5, 2)

        # The native decode against the Python decode on the identical head tensors.
        canvas = np.zeros((model.size, model.size, 3), np.uint8)
        scale = letterbox_into(img, canvas)
        assert scale == D.letterbox(img, model.size)[1]
        heads = run_loom(canvas, [])
        want = D.postprocess(heads_to_outputs(heads, 1, model.size)[0], scale, model.size)
        check("native decode == Python decode on the same heads (to 1e-2 px)",
              same_detections((det, kps), want, box_tol=1e-2, score_tol=1e-4, kps_tol=1e-2))

        # Four distinct images through two resident calls, each graded against
        # onnxruntime on its own letterboxed input; a replicated or reordered
        # batch would fail the cross-image check.
        h, w = img.shape[:2]
        images = [img, img[:, ::-1].copy(), img[h // 4:, w // 4:].copy(), img[::-1].copy()]
        results = model.detect_batch(images)
        sess = ort.InferenceSession(str(G.model_path()), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        counts = []
        for i, (image, got) in enumerate(zip(images, results)):
            det_img, s = D.letterbox(image, model.size)
            want = D.postprocess(sess.run(None, {name: D.blob(det_img)}), s, model.size)
            check(f"batch image {i}: {len(got[0])} faces (onnxruntime {len(want[0])})", same_detections(got, want))
            counts.append(len(got[0]))
        check(f"images are distinct: face counts {counts}",
              not all(np.array_equal(results[0][0], r[0]) for r in results[1:]))
        first = model.detect(images[0])
        saved = (first[0].copy(), first[1].copy())
        model.detect(images[1])
        check("a later call does not mutate an earlier result",
              np.array_equal(first[0], saved[0]) and np.array_equal(first[1], saved[1]))
        check("max_num caps the result", len(model.detect(img, max_num=2)[0]) == 2)

        # A caller's own letterbox: the same canvas must reproduce detect() bitwise,
        # a stacked array is accepted in place, and det_scales=None means canvas pixels.
        det_img, s = D.letterbox(img, model.size)
        (lb_det, lb_kps), = model.detect_letterboxed([det_img], [s])
        check("detect_letterboxed on insightface's own canvas == detect()",
              np.array_equal(lb_det, det) and np.array_equal(lb_kps, kps))
        stacked = np.stack([det_img, D.letterbox(images[1], model.size)[0]])
        two = model.detect_letterboxed(stacked, [s, D.letterbox(images[1], model.size)[1]])
        check("stacked canvases through one call match detect_batch",
              all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(two, results[:2])))
        (cv_det, cv_kps), = model.detect_letterboxed([det_img])
        check("det_scales=None returns canvas pixels",
              np.allclose(cv_det[:, :4], det[:, :4] * s, atol=1e-3) and np.allclose(cv_kps, kps * s, atol=1e-3))
        area = cv2.resize(img, (det_img.shape[1], int(round(img.shape[0] * det_img.shape[1] / img.shape[1]))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros_like(det_img); canvas[:area.shape[0]] = area
        (ar_det, _), = model.detect_letterboxed([canvas], [s])
        check("a different resampler still finds the faces", len(ar_det) == len(det))
        require_raises(ValueError, "uint8", lambda: model.detect_letterboxed([det_img.astype(np.float32)]))
        require_raises(ValueError, "det_scales", lambda: model.detect_letterboxed([det_img], [s, s]))

        # The ABI rejects wrong sizes before launching anything.
        error = ctypes.create_string_buffer(1024)
        u8, f32, i32 = (ctypes.POINTER(t) for t in (ctypes.c_uint8, ctypes.c_float, ctypes.c_int32))
        inp, cand, cnt = model._input[:1], model._candidates[:1], model._counts[:1]
        args = lambda nb, nc, ncount, thresh=0.5: (model._handle, inp.ctypes.data_as(u8), nb, 1, thresh,
                                                    cand.ctypes.data_as(f32), nc, cnt.ctypes.data_as(i32), ncount,
                                                    error, len(error))
        status = model._native.scrfd_run(*args(inp.nbytes - 1, cand.size, 1))
        check("ABI rejects a short input", status == 64 and b"input has" in error.value)
        status = model._native.scrfd_run(*args(inp.nbytes, cand.size - 1, 1))
        check("ABI rejects a misshapen candidate buffer", status == 64 and b"candidates has" in error.value)
        status = model._native.scrfd_run(*args(inp.nbytes, cand.size, 2))
        check("ABI rejects a wrong counts length", status == 64 and b"counts has" in error.value)
        status = model._native.scrfd_run(*args(inp.nbytes, cand.size, 1, 1.0))
        check("ABI rejects det_thresh outside (0, 1)", status == 64 and b"det_thresh" in error.value)
        status = model._native.scrfd_run(*args(inp.nbytes, CANDIDATE_FLOATS, 1, 0.01))
        check("candidate overflow is an error, not a truncation", status == 1 and b"max_candidates" in error.value)

        # The Python lock protects the shared buffers while ctypes drops the GIL.
        want_t = [model.detect(images[i]) for i in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            threaded = list(pool.map(model.detect, images[:2]))
        check("threads see the same results as sequential calls",
              all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(threaded, want_t)))

        # A second session coexists; closing it leaves the first intact.
        with SCRFDLoom(max_batch=1) as other:
            check("a second session agrees", same_detections(other.detect(img), fixture))
        check("the first session survives the second's close", same_detections(model.detect(img), fixture))

    assert model.closed
    model.close()
    require_raises(SCRFDError, "closed", lambda: model.detect(img))
    check("closed sessions refuse calls", True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
