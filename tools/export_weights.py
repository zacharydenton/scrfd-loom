"""Export det_10g's weights in the layout the Loom kernels read.

Every conv becomes one f16 matrix W[Cout_pad][K_pad] in the implicit-GEMM gather
order  k = (dy*3 + dx) * Cin_pad + c  (ONNX stores [Cout][Cin][kh][kw]), plus an
f32 bias [Cout_pad]. Cin is padded to a multiple of 4 so a vector<4xf16> gather
never straddles a tap; K to a multiple of 32 for the k-loop; Cout to a multiple
of 64 so both the 64- and 32-wide N-tile kernels can read the same file.
Padding is zeros in both the weights and the bias, so results are unchanged.

The three head convs per level (score 2, box 8, kps 20) are fused into one conv
of Cout 30 -> 64, in that channel order, which is exactly insightface's
anchor-major row order once the NHWC output is reshaped. The scalar Mul that
follows each box conv in the graph is folded into that conv's weights and bias.

Output: build/weights/{weights_f16.bin, manifest_f16.txt, weights.bin,
manifest.txt}, manifest lines `name offset count` in elements, as dinov3-loom.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G

ROOT = Path(__file__).resolve().parent.parent
CIN_ALIGN, K_ALIGN, COUT_ALIGN = 4, 32, 64


def align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


# Which convs run on the 32-wide N tile. A conv whose Cout is listed here writes
# align32(Cout) columns instead of align64(Cout); everything downstream -- the
# consumer's row stride, the 1x1 convs' K, the weight padding, the host's grid --
# follows from tile_for(), so the policy is decided in exactly one place. Set by
# measurement (tools/ab_conv.py); empty means every layer uses the 64-wide tile.
NARROW_COUT: frozenset[int] = frozenset()


def tile_for(cout: int) -> int:
    return 32 if cout in NARROW_COUT else COUT_ALIGN


def storage_stride(channels: int) -> int:
    """Physical channel stride of an NHWC activation: every conv writes its
    padded Cout, so that is the width its consumers see; the converted stem
    image is the one 4-wide tensor."""
    return 4 if channels <= 4 else align(channels, tile_for(channels))


def pack(weight: np.ndarray, bias: np.ndarray, cout_align: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """[Cout][Cin][k][k], [Cout] -> W16[Cout_pad][K_pad], b32[Cout_pad], shape info.

    cout_align defaults to the tile policy; the A/B harness overrides it to pack
    the same weights for both tiles.
    """
    cout, cin, k, _ = weight.shape
    cout_pad = align(cout, cout_align or tile_for(cout))
    # A 3x3 conv gathers c < cin_pad (4-aligned) inside rows of the input's
    # storage stride. A 1x1 conv is the plain matmul, whose A *is* the input
    # tensor, so its K must equal that stride and the weights are zero beyond cin.
    cin_pad = align(cin, CIN_ALIGN) if k == 3 else storage_stride(cin)
    k_pad = align(k * k * cin_pad, K_ALIGN)
    w = np.zeros((cout_pad, k * k, cin_pad), np.float32)
    w[:cout, :, :cin] = weight.transpose(0, 2, 3, 1).reshape(cout, k * k, cin)   # [co][dy*k+dx][c]
    w = w.reshape(cout_pad, k * k * cin_pad)
    w = np.concatenate([w, np.zeros((cout_pad, k_pad - w.shape[1]), np.float32)], axis=1)
    b = np.zeros(cout_pad, np.float32)
    b[:cout] = bias
    info = dict(cout=cout, cin=cin, k=k, cin_pad=cin_pad, cin_stride=storage_stride(cin),
                k_pad=k_pad, cout_pad=cout_pad)
    return w.astype(np.float16), b, info


def head_groups(graph: G.Graph) -> dict[str, tuple[G.Op, G.Op, G.Op, float]]:
    """stride -> (score conv, box conv, kps conv, box Mul scale) for each level."""
    groups = {}
    for op in graph.convs:
        if op.cout not in (2, 8, 20):
            continue
        src = op.inputs[0]
        level = groups.setdefault(src, {})
        level[op.cout] = op
    out = {}
    for src, level in groups.items():
        score, box, kps = level[2], level[8], level[20]
        mul = next(o for o in graph.consumers(box.output) if o.kind == "mul")
        stride = graph.size // score.out_shape[2]
        out[f"h{stride}"] = (score, box, kps, mul.scale)
    assert sorted(out) == ["h16", "h32", "h8"], sorted(out)
    return out


def fused_head(score: G.Op, box: G.Op, kps: G.Op, scale: float) -> tuple[np.ndarray, np.ndarray]:
    w = np.concatenate([score.weight, box.weight * scale, kps.weight], axis=0)
    b = np.concatenate([score.bias, box.bias * scale, kps.bias])
    return w, b


def main() -> None:
    graph = G.load()
    out_dir = ROOT / "build/weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    f16: list[np.ndarray] = []; f32: list[np.ndarray] = []
    man16: list[str] = []; man32: list[str] = []
    off16 = off32 = 0
    shapes: list[str] = []

    def emit(name: str, w16: np.ndarray, b32: np.ndarray, info: dict) -> None:
        nonlocal off16, off32
        f16.append(w16.ravel()); man16.append(f"{name} {off16} {w16.size}"); off16 += w16.size
        f32.append(b32.ravel()); man32.append(f"{name}_b {off32} {b32.size}"); off32 += b32.size
        shapes.append(f"{name} " + " ".join(f"{k}={v}" for k, v in info.items()))

    heads = head_groups(graph)
    head_members = {op.name for group in heads.values() for op in group[:3]}
    for op in graph.convs:
        if op.name in head_members:
            continue
        emit(op.name, *pack(op.weight, op.bias))
    for name, (score, box, kps, scale) in sorted(heads.items(), key=lambda kv: int(kv[0][1:])):
        w, b = fused_head(score, box, kps, scale)
        emit(name, *pack(w, b))

    np.concatenate(f16).astype(np.float16).tofile(out_dir / "weights_f16.bin")
    np.concatenate(f32).astype(np.float32).tofile(out_dir / "weights.bin")
    (out_dir / "manifest_f16.txt").write_text("\n".join(man16) + "\n")
    (out_dir / "manifest.txt").write_text("\n".join(man32) + "\n")
    (out_dir / "shapes.txt").write_text("\n".join(shapes) + "\n")
    print(f"{len(man16)} weight matrices, {off16 * 2 / 1e6:.1f} MB f16; {len(man32)} biases, "
          f"{off32 * 4 / 1e3:.0f} KB f32 -> {out_dir}")


if __name__ == "__main__":
    main()
