"""insightface's SCRFD pre- and post-processing, vendored so the detections this
repo produces are compared against the exact arithmetic production uses.

`letterbox`, `blob`, `postprocess` and `nms` reproduce `SCRFD.detect` ->
`_detect_candidates` -> `forward` -> `nms` from
python-package/insightface/model_zoo/scrfd.py (deepinsight/insightface, MIT
licence, author Jia Guo), for the batched=False, use_kps=True model this repo
implements. `distance2bbox`/`distance2kps`/`nms` are copied verbatim apart from
the unused torch `.clamp` branches.
"""
from __future__ import annotations

import numpy as np

STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
DET_THRESH = 0.5
NMS_THRESH = 0.4
INPUT_MEAN = 127.5
INPUT_STD = 128.0


def letterbox(img_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, float]:
    """Resize to fit, top-left aligned on a black canvas. Returns (det_img, det_scale)."""
    import cv2
    im_ratio = float(img_bgr.shape[0]) / img_bgr.shape[1]
    model_ratio = 1.0
    if im_ratio > model_ratio:
        new_height = size
        new_width = int(new_height / im_ratio)
    else:
        new_width = size
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img_bgr.shape[0]
    resized = cv2.resize(img_bgr, (new_width, new_height))
    det_img = np.zeros((size, size, 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized
    return det_img, det_scale


def blob(det_img_bgr: np.ndarray) -> np.ndarray:
    """cv2.dnn.blobFromImage(img, 1/128, size, (127.5,)*3, swapRB=True): [1,3,S,S] f32 RGB."""
    import cv2
    size = (det_img_bgr.shape[1], det_img_bgr.shape[0])
    return cv2.dnn.blobFromImage(det_img_bgr, 1.0 / INPUT_STD, size,
                                 (INPUT_MEAN, INPUT_MEAN, INPUT_MEAN), swapRB=True)


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def anchor_centers(height: int, width: int, stride: int) -> np.ndarray:
    centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    centers = (centers * stride).reshape((-1, 2))
    return np.stack([centers] * NUM_ANCHORS, axis=1).reshape((-1, 2))


def nms(dets: np.ndarray, thresh: float = NMS_THRESH) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def postprocess(outputs: list[np.ndarray], det_scale: float, size: int = 640,
                det_thresh: float = DET_THRESH, nms_thresh: float = NMS_THRESH,
                max_num: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The nine raw outputs (ONNX order: scores x3, bboxes x3, kps x3) -> (det [n,5], kps [n,5,2]).

    Boxes and keypoints are in original-image pixels, as SCRFD.detect returns them.
    """
    fmc = len(STRIDES)
    scores_list, bboxes_list, kpss_list = [], [], []
    for idx, stride in enumerate(STRIDES):
        scores = outputs[idx]
        bbox_preds = outputs[idx + fmc] * stride
        kps_preds = outputs[idx + fmc * 2] * stride
        height = width = size // stride
        centers = anchor_centers(height, width, stride)
        pos_inds = np.where(scores >= det_thresh)[0]
        bboxes = distance2bbox(centers, bbox_preds)
        scores_list.append(scores[pos_inds])
        bboxes_list.append(bboxes[pos_inds])
        kpss = distance2kps(centers, kps_preds).reshape((-1, 5, 2))
        kpss_list.append(kpss[pos_inds])
    if sum(s.size for s in scores_list) == 0:
        return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
    scores = np.vstack(scores_list)
    order = scores.ravel().argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale
    kpss = np.vstack(kpss_list) / det_scale
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)[order, :]
    kpss = kpss[order, :, :]
    keep = nms(pre_det, nms_thresh)
    det, kpss = pre_det[keep, :], kpss[keep, :, :]
    if max_num > 0 and det.shape[0] > max_num:
        det, kpss = det[:max_num], kpss[:max_num]
    return det, kpss
