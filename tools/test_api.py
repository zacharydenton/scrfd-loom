"""The Python API: SCRFDLoom.detect() against insightface's fixture, and
detect_batch() on distinct images -- each graded against onnxruntime on the same
letterboxed input, plus a cross-image check that fails if the batch were
replicated or reordered."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from scrfd_loom import SCRFDLoom
from test_reference import test_image, iou, FIXTURE
import decode as D
import graph as G


def main() -> int:
    import cv2, onnxruntime as ort
    ok = True
    model = SCRFDLoom(max_batch=4)
    img = cv2.imread(str(test_image()))

    det, kps = model.detect(img)
    fx = json.loads(FIXTURE.read_text())
    det_i, kps_i = np.array(fx["det"], np.float32), np.array(fx["kps"], np.float32)
    same = det.shape == det_i.shape and all(iou(a, b) > 0.99 and abs(a[4] - b[4]) < 1e-2
                                             and np.abs(ka - kb).max() < 1.0
                                             for a, b, ka, kb in zip(det, det_i, kps, kps_i))
    ok &= same
    print(f"  {'PASS' if same else 'FAIL'} detect(): {len(det)} faces vs insightface's {len(det_i)}")

    # Four distinct images: the original, mirrored, a crop, and upside-down.
    h, w = img.shape[:2]
    images = [img, img[:, ::-1].copy(), img[h // 4:, w // 4:].copy(), img[::-1].copy()]
    results = model.detect_batch(images)
    sess = ort.InferenceSession(str(G.model_path()), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    counts = []
    for i, (image, (det_b, kps_b)) in enumerate(zip(images, results)):
        det_img, scale = D.letterbox(image, model.size)
        want, want_k = D.postprocess(sess.run(None, {name: D.blob(det_img)}), scale, model.size)
        match = det_b.shape == want.shape and all(iou(a, b) > 0.99 and abs(a[4] - b[4]) < 1e-2
                                                   and np.abs(ka - kb).max() < 1.0
                                                   for a, b, ka, kb in zip(det_b, want, kps_b, want_k))
        ok &= match
        counts.append(len(det_b))
        print(f"  {'PASS' if match else 'FAIL'} batch image {i}: {len(det_b)} faces (onnxruntime {len(want)})")
    # a replicated or reordered batch would give every image the original's boxes
    distinct = not all(np.array_equal(results[0][0], r[0]) for r in results[1:])
    ok &= distinct
    print(f"  {'PASS' if distinct else 'FAIL'} images are distinct: face counts {counts}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
