"""Small, dependency-light post-processing helpers used by the installed API.

The fuller InsightFace reference implementation remains in ``tools/decode.py``;
this module contains only the pieces needed at runtime so a wheel does not
depend on the repository's development-tools directory.
"""
from __future__ import annotations

import numpy as np

DET_THRESH = 0.5
NMS_THRESH = 0.4


def nms(dets: np.ndarray, thresh: float = NMS_THRESH) -> list[int]:
    """InsightFace's inclusive-coordinate greedy NMS."""
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= thresh)[0] + 1]
    return keep


def limit_detections(
    det: np.ndarray,
    kps: np.ndarray,
    max_num: int,
    metric: str,
    image_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SCRFD.detect's post-NMS max_num selection exactly."""
    if max_num <= 0 or det.shape[0] <= max_num:
        return det, kps

    area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    if metric == "max":
        values = area
    else:
        if image_shape is None:
            raise ValueError("image_shape is required for the default max_num metric")
        height, width = image_shape
        offsets = np.vstack((
            (det[:, 0] + det[:, 2]) / 2 - width // 2,
            (det[:, 1] + det[:, 3]) / 2 - height // 2,
        ))
        values = area - np.sum(np.power(offsets, 2.0), axis=0) * 2.0
    chosen = values.argsort()[::-1][:max_num]
    return det[chosen], kps[chosen]


__all__ = ["DET_THRESH", "NMS_THRESH", "limit_detections", "nms"]
