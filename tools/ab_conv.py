"""Interleaved A/B of two 3x3 conv kernel sources on the real heavy layers.

    python3 tools/ab_conv.py experiments/conv3x3_narrow_f16_wmma.loom [layer ...]

Compares the candidate against kernels/conv3x3_f16_wmma.loom, correctness
first (against the float64 reference, or it does not count), then best-of-N
timing with the two alternated so both see the same machine. The four layers
are the ones the profile is made of: 56->56 @160^2, 88->88 @80^2, 224->224
@20^2 and the 28->56 @320^2 stem conv.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT
import graph as G
import reference as R
from export_weights import pack, storage_stride

BASE = ROOT / "kernels/conv3x3_f16_wmma.loom"
ROUNDS = 5


def symbol_of(path: Path) -> tuple[str, str, int, int]:
    """(export symbol, config namespace, N tile width, gather width). A source whose
    name contains `narrow` is the 64x32 tile and one containing `n128` the 64x128
    tile: n_size is aligned to the tile and the grid is n_size/tile wide. The gather width is read off the cin_pad declaration: a kernel
    that gathers vector<8xf16> declares cin_pad as a multiple of 8."""
    text = path.read_text()
    import re
    m = re.search(r'export\("([a-z0-9_]+)"\)', text)
    sym = m.group(1)
    ns = re.search(r"config\.decl @([a-z0-9_.]+)\.k_size", text).group(1)
    gather = int(re.search(r"\.cin_pad : %value: index where \[range\(%value, (\d+)", text).group(1))
    tile = 128 if "n128" in path.name else 32 if "narrow" in path.name else 64
    return sym, ns, tile, gather


def main() -> int:
    candidate = Path(sys.argv[1]).resolve()
    graph = G.load()
    convs = {op.name: op for op in graph.convs}
    rng = np.random.default_rng(3)
    ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        print(f"{'layer':<26s} {'base us':>9s} {'TF/s':>5s} {'cand us':>9s} {'TF/s':>5s} {'speedup':>8s}")
        layers = sys.argv[2:] or ["c00", "c02", "c03", "c10", "c27"]
        for name, batch in ((n, 4) for n in layers):
            op = convs[name]
            # Pack once per (tile, gather width): the weight matrix follows the
            # kernel's Cout and Cin padding. All hold the same values.
            packed = {(tile, ga): pack(op.weight, op.bias, cout_align=tile, cin_align=ga)
                      for tile in (128, 64, 32) for ga in (4, 8)}
            _, cin, h, w = graph.shapes[op.inputs[0]]
            # rows at least 8 wide so an 8-wide gather stays aligned on the stem's 3 channels
            cs, s = max(storage_stride(cin), 8), op.stride
            ho, wo = h // s, w // s
            m = batch * ho * wo
            x = (rng.standard_normal((batch, cin, h, w)) * 0.5).astype(np.float32)
            x_nhwc = np.zeros((batch, h, w, cs), np.float16)
            x_nhwc[..., :cin] = x.transpose(0, 2, 3, 1).astype(np.float16)
            want = R.conv2d(x_nhwc[..., :cin].astype(np.float64).transpose(0, 3, 1, 2), op.weight, op.bias, s, 1)
            want = want.transpose(0, 2, 3, 1).reshape(m, op.cout)

            built = {}
            for tag, src in (("base", BASE), ("cand", candidate)):
                sym, ns, tile, ga = symbol_of(src)
                n_size = (op.cout + tile - 1) // tile * tile
                info = packed[(tile, ga)][2]
                hs = tmp / f"{tag}_{name}.hsaco"
                compile_kernel(src, sym, {f"{ns}.height": h, f"{ns}.width": w, f"{ns}.stride": s,
                                          f"{ns}.cin_pad": info["cin_pad"], f"{ns}.cin_stride": cs,
                                          f"{ns}.k_size": info["k_pad"], f"{ns}.n_size": n_size}, hs)
                built[tag] = (hs, sym, n_size, tile, ga)
            best = {"base": 1e9, "cand": 1e9}
            for r in range(ROUNDS):
                for tag in ("base", "cand") if r % 2 == 0 else ("cand", "base"):
                    hs, sym, n_size, tile, ga = built[tag]
                    w16, b32, _ = packed[(tile, ga)]
                    args = [("i32", m), ("in_f16", x_nhwc.reshape(-1, cs)), ("in_f16", w16), ("in", b32),
                            ("out_f16", ((m, n_size), np.float16))]
                    grid = (n_size // tile, (m + 63) // 64, 1)
                    (y,), t = launch(hs, sym, grid, (256, 1, 1), args, tmp, repeat=10)
                    if r == 0:
                        ok &= report(f"  {tag} {name} correctness", y[:, :op.cout], want, atol=5e-2, rtol=5e-2)
                    best[tag] = min(best[tag], t["per_launch_us"])
            fl = 2.0 * m * 9 * cin * op.cout
            tf = lambda us: fl / (us * 1e-6) / 1e12
            print(f"{name} {cin:>3d}->{op.cout:<3d} {h:>3d}^2 s{s} B={batch}   {best['base']:9.1f} {tf(best['base']):5.1f} "
                  f"{best['cand']:9.1f} {tf(best['cand']):5.1f} {best['base'] / best['cand']:7.3f}x")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
