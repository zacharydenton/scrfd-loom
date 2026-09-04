"""Float64 NumPy interpreter for the det_10g graph, in NCHW, matching onnxruntime.

Deliberately independent of every kernel and of the export's NHWC layout: this
is the oracle the Loom kernels are graded against, so a layout mistake in the
export cannot be mirrored here. Convolution is explicit im2col + matmul in
float64, which is slow (a few seconds per image) and exactly right.
"""
from __future__ import annotations

import numpy as np

import graph as G


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, pad: int) -> np.ndarray:
    """x [N,C,H,W] f64, w [Cout,Cin,k,k], b [Cout] -> [N,Cout,Ho,Wo] f64."""
    n, c, h, wd = x.shape
    cout, cin, k, _ = w.shape
    assert cin == c
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - k) // stride + 1, (wd + 2 * pad - k) // stride + 1
    # im2col: [N, Ho, Wo, C, k, k] via stride tricks, then one matmul per image
    sN, sC, sH, sW = xp.strides
    cols = np.lib.stride_tricks.as_strided(
        xp, shape=(n, ho, wo, c, k, k),
        strides=(sN, sH * stride, sW * stride, sC, sH, sW), writeable=False)
    cols = cols.reshape(n, ho * wo, c * k * k)
    out = cols @ w.reshape(cout, -1).T.astype(np.float64) + b.astype(np.float64)
    return out.reshape(n, ho, wo, cout).transpose(0, 3, 1, 2)


def pool2(x: np.ndarray, op: str) -> np.ndarray:
    n, c, h, w = x.shape
    v = x.reshape(n, c, h // 2, 2, w // 2, 2)
    return v.max(axis=(3, 5)) if op == "max" else v.mean(axis=(3, 5))


def resize2x(x: np.ndarray) -> np.ndarray:
    return x.repeat(2, axis=2).repeat(2, axis=3)


def flatten_head(x: np.ndarray, width: int) -> np.ndarray:
    """ONNX Transpose(2,3,0,1) then Reshape(-1, width): [N,C,H,W] -> [N*H*W*C/width, width]."""
    return x.transpose(2, 3, 0, 1).reshape(-1, width)


def forward(graph: G.Graph, pixel_values: np.ndarray) -> dict[str, np.ndarray]:
    """pixel_values [N,3,S,S] (already (x-127.5)/128, RGB) -> the nine outputs by ONNX name."""
    t: dict[str, np.ndarray] = {graph.input: np.asarray(pixel_values, dtype=np.float64)}
    for op in graph.ops:
        a = t[op.inputs[0]]
        if op.kind == "conv":
            t[op.output] = conv2d(a, op.weight, op.bias, op.stride, op.pad)
        elif op.kind == "relu":
            t[op.output] = np.maximum(a, 0.0)
        elif op.kind == "sigmoid":
            t[op.output] = 1.0 / (1.0 + np.exp(-a))
        elif op.kind == "add":
            t[op.output] = a + t[op.inputs[1]]
        elif op.kind == "mul":
            t[op.output] = a * op.scale
        elif op.kind == "maxpool":
            t[op.output] = pool2(a, "max")
        elif op.kind == "avgpool":
            t[op.output] = pool2(a, "avg")
        elif op.kind == "resize2x":
            t[op.output] = resize2x(a)
        elif op.kind == "flatten":
            t[op.output] = flatten_head(a, op.width)
        else:
            raise NotImplementedError(op.kind)
    return {name: t[name] for name in graph.outputs}


def blob_from_rgb_f64(image_rgb_u8: np.ndarray) -> np.ndarray:
    """(S,S,3) RGB uint8 -> [1,3,S,S] float64 with insightface's (x-127.5)/128."""
    x = image_rgb_u8.astype(np.float64).transpose(2, 0, 1)[None]
    return (x - 127.5) / 128.0
