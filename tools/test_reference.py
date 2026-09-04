"""The float64 reference vs onnxruntime (CPU, f32) on the nine raw outputs, and the
vendored decode on a real face image.

This is the oracle's own test: everything downstream is graded against
reference.py, so it has to agree with the deployed graph first.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G
import reference as R
import decode as D

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "build/images/t1.jpg"
IMAGE_URL = ("https://raw.githubusercontent.com/deepinsight/insightface/master/"
             "python-package/insightface/data/images/t1.jpg")
# insightface's own SCRFD.detect() on t1.jpg at 640x640, det_thresh 0.5,
# nms_thresh 0.4 -- captured once with the real package, so this test grades
# the vendored decode against production, not against itself.
FIXTURE = ROOT / "tools/fixtures/t1_insightface.json"


def test_image() -> Path:
    if not IMAGE.exists():
        import urllib.request
        IMAGE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMAGE_URL, IMAGE)
    return IMAGE


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def ort_session():
    import onnxruntime as ort
    return ort.InferenceSession(str(G.model_path()), providers=["CPUExecutionProvider"])


def main() -> int:
    import cv2
    ok = True
    graph = G.load()
    sess = ort_session()
    input_name = sess.get_inputs()[0].name

    img = cv2.imread(str(test_image()))
    if img is None:
        print(f"  cannot read test image {IMAGE}"); return 1
    det_img, det_scale = D.letterbox(img, graph.size)
    x = D.blob(det_img)                                  # [1,3,640,640] f32, RGB
    ort_outs = sess.run(None, {input_name: x})

    t = time.perf_counter()
    ref = R.forward(graph, x.astype(np.float64))
    print(f"  reference forward: {time.perf_counter() - t:.1f} s")

    for name, got in zip(graph.outputs, ort_outs):
        want = ref[name]
        if got.shape != want.shape:
            print(f"  FAIL {name}: shape {got.shape} != {want.shape}"); ok = False; continue
        err = np.abs(got.astype(np.float64) - want)
        scale = np.abs(want).max() + 1e-6
        rel = err.max() / scale
        good = rel < 1e-4
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} {name:>4s} {str(got.shape):>12s}  max_abs={err.max():.2e}  "
              f"rel_to_max={rel:.2e}")

    # The vendored decode, on ORT's outputs, against insightface's own detections.
    import json
    fixture = json.loads(FIXTURE.read_text())
    det_i, kps_i = np.array(fixture["det"], np.float32), np.array(fixture["kps"], np.float32)
    det, kps = D.postprocess(ort_outs, det_scale, graph.size)
    same = det.shape == det_i.shape and kps.shape == kps_i.shape
    if same:
        for a, b, ka, kb in zip(det, det_i, kps, kps_i):
            same &= iou(a, b) > 0.99 and abs(a[4] - b[4]) < 1e-2 and np.abs(ka - kb).max() < 1.0
    ok &= same
    print(f"  {'PASS' if same else 'FAIL'} decode vs insightface: {len(det)} faces on {IMAGE.name} "
          f"(insightface: {len(det_i)}); scores {np.round(det[:, 4], 3).tolist()[:6]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
