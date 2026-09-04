"""End-to-end: the Loom runner vs onnxruntime and vs insightface's detections.

Two gates, in the order the plan states them:
  1. the nine raw outputs (scores post-sigmoid, box and keypoint distances) vs
     onnxruntime CPU, cosine and max-abs per output;
  2. decoded detections vs the fixture of insightface's own SCRFD.detect() on the
     same image: IoU > 0.99, score delta < 1e-2, keypoints < 1 px.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G
import decode as D
from test_reference import test_image, iou, FIXTURE

ROOT = Path(__file__).resolve().parent.parent
STRIDES = (8, 16, 32)


def heads_to_outputs(heads: list[np.ndarray], batch: int, size: int) -> list[list[np.ndarray]]:
    """Three [B*H*W][64] f16 head tensors -> per image, the nine ONNX outputs.

    Fused channel order per pixel: [score(a0,a1) | box(a0: 4, a1: 4) | kps(a0: 10, a1: 10)].
    ONNX rows are pixel-major, anchor-minor: row = pixel*2 + anchor.
    """
    per_image = [[] for _ in range(batch)]
    parts = {"score": [], "box": [], "kps": []}
    for level, stride in enumerate(STRIDES):
        hw = (size // stride) ** 2
        x = heads[level].astype(np.float32).reshape(batch, hw, 64)
        score = 1.0 / (1.0 + np.exp(-x[:, :, 0:2]))                  # (B, HW, 2)
        box = x[:, :, 2:10].reshape(batch, hw, 2, 4)
        kps = x[:, :, 10:30].reshape(batch, hw, 2, 10)
        parts["score"].append(score.reshape(batch, hw * 2, 1))
        parts["box"].append(box.reshape(batch, hw * 2, 4))
        parts["kps"].append(kps.reshape(batch, hw * 2, 10))
    for b in range(batch):
        per_image[b] = [parts["score"][l][b] for l in range(3)] + [parts["box"][l][b] for l in range(3)] \
                       + [parts["kps"][l][b] for l in range(3)]
    return per_image


def run_loom(det_img: np.ndarray, extra_args: list[str]) -> list[np.ndarray]:
    """Letterboxed BGR uint8 image(s), (S,S,3) or (B,S,S,3) -> the three head tensors, via the CLI."""
    image = np.ascontiguousarray(det_img, dtype=np.uint8)
    if image.ndim == 3:
        image = image[None]
    batch, size = image.shape[0], image.shape[1]
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "images.bin", Path(tmp) / "heads.bin"
        image.tofile(src)
        subprocess.run([str(ROOT / "host/scrfd"), "--weights", str(ROOT / "build/weights"),
                        "--kernels", str(ROOT / "build/kernels"), "--input", str(src),
                        "--output", str(dst), "--batch", str(batch)] + extra_args,
                       check=True, cwd=ROOT, env=env, capture_output=True)
        raw = np.fromfile(dst, dtype=np.float16)
    heads, off = [], 0
    for stride in STRIDES:
        n = batch * (size // stride) ** 2 * 64
        heads.append(raw[off:off + n]); off += n
    assert off == raw.size, (off, raw.size)
    return heads


def cosine(a, b) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    import cv2
    import onnxruntime as ort
    ok = True
    graph = G.load()
    img = cv2.imread(str(test_image()))
    det_img, det_scale = D.letterbox(img, graph.size)
    blob = D.blob(det_img)

    sess = ort.InferenceSession(str(G.model_path()), providers=["CPUExecutionProvider"])
    want = sess.run(None, {sess.get_inputs()[0].name: blob})
    got = heads_to_outputs(run_loom(det_img, sys.argv[1:]), 1, graph.size)[0]

    labels = [f"score/{s}" for s in STRIDES] + [f"box/{s}" for s in STRIDES] + [f"kps/{s}" for s in STRIDES]
    for label, g, w in zip(labels, got, want):
        c = cosine(g, w); err = np.abs(g.astype(np.float64) - w).max()
        good = c > 0.9999 and (err < 2e-2 if label.startswith("score") else err < 0.15)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} {label:<9s} {str(g.shape):>12s} cosine={c:.7f} max_abs={err:.3e}")

    fixture = json.loads(FIXTURE.read_text())
    det_i, kps_i = np.array(fixture["det"], np.float32), np.array(fixture["kps"], np.float32)
    det, kps = D.postprocess(got, det_scale, graph.size)
    same = det.shape == det_i.shape and kps.shape == kps_i.shape
    worst = (0.0, 0.0, 0.0)
    if same:
        for a, b, ka, kb in zip(det, det_i, kps, kps_i):
            j, ds, dk = iou(a, b), abs(a[4] - b[4]), np.abs(ka - kb).max()
            worst = (max(worst[0], 1 - j), max(worst[1], ds), max(worst[2], dk))
            same &= j > 0.99 and ds < 1e-2 and dk < 1.0
    ok &= same
    print(f"  {'PASS' if same else 'FAIL'} detections vs insightface: {len(det)} faces (insightface {len(det_i)}); "
          f"worst 1-IoU={worst[0]:.4f} score delta={worst[1]:.4f} kps={worst[2]:.2f} px; "
          f"scores {np.round(det[:, 4], 3).tolist()[:6]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
