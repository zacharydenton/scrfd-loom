"""Interleaved benchmark: the Loom detector vs onnxruntime+MIGraphX on the same GPU.

The deployed ONNX graph is batch-1-only, so MIGraphX has exactly one number:
ms per image at batch 1. Ours is measured at batch 1 and at the batch sizes the
runner supports. Rounds are interleaved and the best of N is kept per
configuration, which is the honest protocol under contention; the machine's
CPU load is recorded next to the numbers because it has been high throughout.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G
import decode as D
from test_reference import test_image

ROOT = Path(__file__).resolve().parent.parent
ROUNDS = 3
BATCHES = (1, 8, 32)


def loom_img_per_s(blob_path: Path, batch: int, images: int = 96) -> float:
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    out = subprocess.run([str(ROOT / "host/scrfd"), "--input", str(blob_path), "--batch", str(batch),
                          "--repeat", str(max(2, images // batch))], check=True, capture_output=True,
                         text=True, cwd=ROOT, env=env).stdout.strip().splitlines()[-1]
    return json.loads(out)["img_per_s"]


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
    det_img, _ = D.letterbox(img, G.INPUT_SIZE)
    blob = D.blob(det_img)
    blob_path = ROOT / "build/blob_t1.bin"
    blob.astype(np.float32).tofile(blob_path)

    load = os.getloadavg()[0]
    print(f"det_10g (SCRFD-10GF), 640x640 -- gfx1151; CPU load average {load:.1f}; best of {ROUNDS} interleaved rounds\n")
    best: dict[str, float] = {}
    for r in range(ROUNDS):
        for b in BATCHES:
            key = f"loom (batch {b})"
            best[key] = max(best.get(key, 0.0), loom_img_per_s(blob_path, b))
        ms = migraphx_ms(blob)
        best["onnxruntime MIGraphX (batch 1)"] = max(best.get("onnxruntime MIGraphX (batch 1)", 0.0), 1000 / ms)
        print(f"  round {r + 1}/{ROUNDS} done", flush=True)

    print(f"\n{'configuration':<36s} {'img/s':>8s} {'ms/img':>8s}   vs MIGraphX")
    ref = best["onnxruntime MIGraphX (batch 1)"]
    for name, v in sorted(best.items(), key=lambda kv: -kv[1]):
        marker = "  <-- this repo" if name.startswith("loom") else ""
        print(f"{name:<36s} {v:8.1f} {1000 / v:8.2f}   {v / ref:5.2f}x{marker}")


if __name__ == "__main__":
    main()
