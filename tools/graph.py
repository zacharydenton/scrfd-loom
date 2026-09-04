"""The det_10g graph, read from the ONNX file and resolved at one input size.

This is the single source of truth: the float64 reference interprets it, the
weight export walks it, and the host's launch table is generated from it, so
the 58 convolutions cannot drift between the three.

Everything that exists only because the ONNX export has a dynamic H/W (Shape,
Gather, Unsqueeze, Slice, Concat feeding Resize) is folded here at a fixed size:
the two Resizes become exact nearest 2x upsamples. The trailing
Transpose+Reshape pairs that flatten each head are kept as one `flatten` op so
the reference produces the nine outputs in the exact ONNX layout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path("~/.insightface/models/buffalo_l/det_10g.onnx").expanduser()
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
# Per anchor: 1 score, 4 box distances, 10 keypoint offsets.
HEAD_CHANNELS = (1, 4, 10)


def model_path() -> Path:
    return Path(os.environ.get("SCRFD_ONNX", DEFAULT_MODEL))


@dataclass
class Op:
    kind: str                 # conv, relu, add, maxpool, avgpool, resize2x, sigmoid, mul, flatten
    name: str                 # conv ops: c00..c57; others: the ONNX output tensor name
    inputs: list[str]
    output: str
    # conv only
    weight: np.ndarray | None = None      # [Cout][Cin][k][k] f32, as exported by torch
    bias: np.ndarray | None = None        # [Cout] f32
    stride: int = 1
    pad: int = 0
    # mul only
    scale: float | None = None
    # flatten only: Transpose(2,3,0,1) + Reshape(-1, width)
    width: int = 0
    # shapes resolved at the fixed input size, NCHW
    out_shape: tuple[int, ...] = field(default=())

    @property
    def cout(self) -> int: return int(self.weight.shape[0])
    @property
    def cin(self) -> int: return int(self.weight.shape[1])
    @property
    def ksize(self) -> int: return int(self.weight.shape[2])


@dataclass
class Graph:
    ops: list[Op]
    input: str
    outputs: list[str]                 # the nine ONNX output names, in ONNX order
    shapes: dict[str, tuple[int, ...]]  # every tensor, NCHW at the fixed size
    size: int

    @property
    def convs(self) -> list[Op]:
        return [op for op in self.ops if op.kind == "conv"]

    def by_output(self, name: str) -> Op:
        return next(op for op in self.ops if op.output == name)

    def consumers(self, name: str) -> list[Op]:
        return [op for op in self.ops if name in op.inputs]


def load(size: int = INPUT_SIZE, path: Path | None = None) -> Graph:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path or model_path()))
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    attr = lambda n: {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}

    shapes: dict[str, tuple[int, ...]] = {g.input[0].name: (1, 3, size, size)}
    ops: list[Op] = []
    conv_index = 0
    # Resolved values of the shape-plumbing tensors, so Resize can be folded.
    folded: dict[str, np.ndarray] = {}

    for n in g.node:
        t = n.op_type
        a = attr(n)
        ins = [x for x in n.input if x not in init]
        out = n.output[0]

        if t == "Conv":
            w, b = init[n.input[1]], init[n.input[2]]
            s, p, k = a.get("strides", [1])[0], a.get("pads", [0])[0], w.shape[2]
            assert a.get("group", 1) == 1 and a.get("dilations", [1])[0] == 1
            N, C, H, W = shapes[ins[0]]
            assert C == w.shape[1], (n.name, C, w.shape)
            ho, wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
            shapes[out] = (N, int(w.shape[0]), ho, wo)
            ops.append(Op("conv", f"c{conv_index:02d}", ins, out, weight=w.astype(np.float32),
                          bias=b.astype(np.float32), stride=s, pad=p, out_shape=shapes[out]))
            conv_index += 1
        elif t in ("Relu", "Sigmoid"):
            shapes[out] = shapes[ins[0]]
            ops.append(Op(t.lower(), out, ins, out, out_shape=shapes[out]))
        elif t == "Add":
            assert shapes[ins[0]] == shapes[ins[1]], (out, shapes[ins[0]], shapes[ins[1]])
            shapes[out] = shapes[ins[0]]
            ops.append(Op("add", out, ins, out, out_shape=shapes[out]))
        elif t == "Mul":
            const = init[n.input[1]]
            assert const.size == 1, f"Mul {out}: expected a scalar, got {const.shape}"
            shapes[out] = shapes[ins[0]]
            ops.append(Op("mul", out, ins, out, scale=float(const.reshape(-1)[0]), out_shape=shapes[out]))
        elif t in ("MaxPool", "AveragePool"):
            k, s, p = a["kernel_shape"][0], a.get("strides", [1])[0], a.get("pads", [0])[0]
            assert (k, s, p) == (2, 2, 0), (t, k, s, p)
            N, C, H, W = shapes[ins[0]]
            shapes[out] = (N, C, H // 2, W // 2)
            ops.append(Op("maxpool" if t == "MaxPool" else "avgpool", out, ins, out, out_shape=shapes[out]))
        elif t == "Resize":
            assert a.get("mode") == b"nearest" and a.get("coordinate_transformation_mode") == b"asymmetric" \
                and a.get("nearest_mode") == b"floor", a
            N, C, H, W = shapes[ins[0]]
            target = folded[n.input[3] if len(n.input) > 3 else n.input[1]]
            assert tuple(int(v) for v in target) == (N, C, 2 * H, 2 * W), (out, target, shapes[ins[0]])
            shapes[out] = (N, C, 2 * H, 2 * W)
            ops.append(Op("resize2x", out, [ins[0]], out, out_shape=shapes[out]))
        elif t == "Shape":
            folded[out] = np.array(shapes[ins[0]], dtype=np.int64)
        elif t == "Gather":
            folded[out] = folded[ins[0]][init[n.input[1]]]
        elif t == "Unsqueeze":
            folded[out] = np.atleast_1d(folded[ins[0]])
        elif t == "Slice":
            st, en = int(init[n.input[1]][0]), int(init[n.input[2]][0])
            folded[out] = folded[ins[0]][st:en]
        elif t == "Concat":
            if all(x in folded for x in ins):
                folded[out] = np.concatenate([np.atleast_1d(folded[x]) for x in ins])
            else:
                raise NotImplementedError("tensor Concat is not in this graph")
        elif t == "Transpose":
            assert list(a["perm"]) == [2, 3, 0, 1], a
            # Paired with the Reshape that follows; recorded when it arrives.
            shapes[out] = shapes[ins[0]]
            folded[out] = np.array([-1], dtype=np.int64)   # marker: "transposed"
        elif t == "Reshape":
            src = ins[0]
            assert src in folded and folded[src].tolist() == [-1], "Reshape not preceded by the head Transpose"
            width = int(init[n.input[1]][1])
            tensor = g.node[[x.output[0] for x in g.node].index(src)].input[0]
            N, C, H, W = shapes[tensor]
            shapes[out] = (N * H * W * C // width, width)
            ops.append(Op("flatten", out, [tensor], out, width=width, out_shape=shapes[out]))
        else:
            raise NotImplementedError(t)

    return Graph(ops=ops, input=g.input[0].name, outputs=[o.name for o in g.output],
                 shapes=shapes, size=size)


if __name__ == "__main__":
    graph = load()
    print(f"{len(graph.ops)} ops, {len(graph.convs)} convs at {graph.size}x{graph.size}")
    for op in graph.ops:
        detail = ""
        if op.kind == "conv":
            detail = f"{op.cin:>3d}->{op.cout:<3d} k{op.ksize} s{op.stride}"
        elif op.kind == "mul":
            detail = f"x {op.scale:.6f}"
        elif op.kind == "flatten":
            detail = f"width {op.width}"
        print(f"  {op.name:>6s} {op.kind:<9s} {str(op.inputs):<22s} -> {op.output:<5s} {str(op.out_shape):<22s} {detail}")
