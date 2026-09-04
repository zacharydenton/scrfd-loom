"""Interleaved benchmark: the Python API end to end vs onnxruntime+MIGraphX.

What is timed for this repo is SCRFDLoom.detect_batch on real images: letterbox,
upload, the network, the native decode and NMS, i.e. what a caller pays. The
deployed ONNX graph is batch-1-only, so MIGraphX has one number, and it is the
network alone (session.run on a prepared blob; its own decode would come on top).
Rounds are interleaved and the best of N kept per configuration, the honest
protocol under contention; the CPU load is recorded next to the numbers.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import graph as G
import decode as D
from test_reference import test_image
from scrfd_loom import SCRFDLoom

ROUNDS = 3
BATCHES = (1, 8, 16, 32)


def loom_img_per_s(model: SCRFDLoom, images: list[np.ndarray], batch: int, total: int = 96) -> float:
    chunk = images[:batch]
    model.detect_batch(chunk)
    calls = max(2, total // batch)
    t = time.perf_counter()
    for _ in range(calls):
        model.detect_batch(chunk)
    return calls * batch / (time.perf_counter() - t)


def migraphx_ms(blob: np.ndarray, steps: int = 30) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(G.model_path()),
                                providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"])
    assert sess.get_providers()[0] == "MIGraphXExecutionProvider", sess.get_providers()
    name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {name: blob})
    t = time.perf_counter()
    for _ in range(steps):
        sess.run(None, {name: blob})
    return (time.perf_counter() - t) / steps * 1000


def main() -> None:
    import cv2
    img = cv2.imread(str(test_image()))
    # distinct images so nothing can be cached across the batch
    h, w = img.shape[:2]
    images = [img, img[:, ::-1].copy(), img[::-1].copy(), img[h // 8:, w // 8:].copy()] * 8
    blob = D.blob(D.letterbox(img, G.INPUT_SIZE)[0])
    model = SCRFDLoom(max_batch=max(BATCHES))

    load = os.getloadavg()[0]
    print(f"det_10g (SCRFD-10GF), {img.shape[1]}x{img.shape[0]} images letterboxed to 640 -- gfx1151; "
          f"CPU load average {load:.1f}; best of {ROUNDS} interleaved rounds\n")
    best: dict[str, float] = {}
    for r in range(ROUNDS):
        for b in BATCHES:
            key = f"scrfd-loom detect_batch (batch {b})"
            best[key] = max(best.get(key, 0.0), loom_img_per_s(model, images, b))
        ms = migraphx_ms(blob)
        key = "onnxruntime MIGraphX, network only (batch 1)"
        best[key] = max(best.get(key, 0.0), 1000 / ms)
        print(f"  round {r + 1}/{ROUNDS} done", flush=True)
    model.close()

    print(f"\n{'configuration':<48s} {'img/s':>8s} {'ms/img':>8s}   vs MIGraphX")
    ref = best["onnxruntime MIGraphX, network only (batch 1)"]
    for name, v in sorted(best.items(), key=lambda kv: -kv[1]):
        marker = "  <-- this repo" if name.startswith("scrfd-loom") else ""
        print(f"{name:<48s} {v:8.1f} {1000 / v:8.2f}   {v / ref:5.2f}x{marker}")


if __name__ == "__main__":
    main()
