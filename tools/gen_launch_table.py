"""Turn the graph into the host's launch schedule, so the 58 convolutions are
never transcribed by hand.

From `graph.py`'s op list this produces:

  host/graph_table.inc      the ordered launch table as C struct literals
  build/launch_table.json   the same, for the Python tests
  scripts/kernels.generated the build lines for every distinct (kernel, config)

What the schedule folds away, in the order the plan states it:

  Relu     -> the `relu` / `relu_add` variant of the conv that produces its input.
  Add      -> hosted on the producer of its *first* input as the `add` /
              `relu_add` variant (a 3x3 conv) or `add_resized` (a lateral 1x1
              whose second input is a nearest-2x Resize). That producer is
              scheduled after the tensor it adds, so the shortcut branch of a
              downsample block (AveragePool -> 1x1) runs before the 3x3 that
              hosts the add. Zero Add launches, and the Resize never exists.
  Mul      -> folded into the box head's weights by the export.
  heads    -> the nine head convs become three fused convs (Cout 30 -> 64).
  Sigmoid, Transpose, Reshape -> the Python API, on the three head buffers.

Buffers are assigned by liveness: a tensor's buffer is free after its last
consumer, and freed buffers are reused by size, so the resident set is a dozen
buffers rather than one per tensor.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G
from export_weights import align, storage_stride, head_groups, CIN_ALIGN, K_ALIGN, COUT_ALIGN

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Launch:
    kind: str                 # convert, conv3x3, matmul, pool
    variant: str              # conv3x3: plain|relu|add|relu_add; matmul: plain|add_resized; pool: max|mean
    name: str                 # weight name (convs) or a label
    src: str                  # input tensor
    dst: str                  # output tensor
    extra: str = ""           # residual (add) or coarse (add_resized) tensor
    stage: str = ""           # profiler label
    # geometry of the *input* for conv/pool, and of the output plane
    h: int = 0
    w: int = 0
    stride: int = 1
    cin: int = 0
    cin_pad: int = 0
    cin_stride: int = 0
    k_size: int = 0
    cout: int = 0
    n_size: int = 0
    ho: int = 0
    wo: int = 0
    src_buf: int = -1
    dst_buf: int = -1
    extra_buf: int = -1

    @property
    def stem(self) -> str:
        """The HSACO this launch runs: one per distinct kernel+config."""
        if self.kind == "convert":
            return "nchw_to_nhwc"
        if self.kind == "pool":
            return f"pool2_{self.variant}_{self.h}x{self.w}_c{self.cin_stride}"
        if self.kind == "matmul":
            geo = f"_{self.h}x{self.w}" if self.variant == "add_resized" else ""
            return f"matmul_{self.variant}_k{self.k_size}_n{self.n_size}{geo}"
        return (f"conv3x3_{self.variant}_{self.h}x{self.w}_s{self.stride}"
                f"_c{self.cin_pad}of{self.cin_stride}_k{self.k_size}_n{self.n_size}")


def build_schedule(graph: G.Graph):
    """Returns (ordered launches, tensor -> (H, W, storage stride), buffer bytes per image)."""
    heads = head_groups(graph)
    head_member = {op.name: level for level, (s, b, k, _) in heads.items() for op in (s, b, k)}

    # --- which op consumes what, for folding --------------------------------------
    producer = {op.output: op for op in graph.ops}
    consumers: dict[str, list[G.Op]] = {}
    for op in graph.ops:
        for t in op.inputs:
            consumers.setdefault(t, []).append(op)

    # alias: tensor name -> the tensor whose buffer actually holds it (after folding)
    alias: dict[str, str] = {}
    def resolve(t: str) -> str:
        while t in alias:
            t = alias[t]
        return t

    launches: list[Launch] = []
    shapes: dict[str, tuple[int, int, int]] = {}   # tensor -> (H, W, stride)

    # the converted input
    _, _, H, W = graph.shapes[graph.input]
    conv_in = "nhwc_input"
    shapes[conv_in] = (H, W, 4)
    launches.append(Launch("convert", "", "convert", graph.input, conv_in, stage="convert",
                           h=H, w=W, cin=3, cin_pad=4, cin_stride=4, ho=H, wo=W))
    alias[graph.input] = conv_in

    fused_relu: set[str] = set()   # Relu ops folded into a conv
    fused_add: set[str] = set()    # Add ops folded into a conv
    for op in graph.ops:
        if op.kind == "conv":
            if op.name in head_member:
                continue                       # emitted once per level below
            variant = "plain"
            extra = ""
            out = op.output
            # An Add whose first input is this conv's output is hosted here.
            adds = [c for c in consumers.get(op.output, []) if c.kind == "add" and c.inputs[0] == op.output]
            if adds:
                add = adds[0]
                other = producer[add.inputs[1]]
                if other.kind == "resize2x":
                    variant, extra = "add_resized", other.inputs[0]
                else:
                    variant, extra = "add", add.inputs[1]
                fused_add.add(add.output)
                out = add.output
            relus = [c for c in consumers.get(out, []) if c.kind == "relu"]
            if relus:
                assert variant != "add_resized", "no ReLU follows an FPN add in this graph"
                variant = "relu" if variant == "plain" else "relu_add"
                fused_relu.add(relus[0].output)
                alias[relus[0].output] = op.output
            if adds:
                alias[add.output] = op.output
            _, cin, h, w = graph.shapes[op.inputs[0]]
            cs = storage_stride(cin)
            kind = "conv3x3" if op.ksize == 3 else "matmul"
            cin_pad = align(cin, CIN_ALIGN) if op.ksize == 3 else cs
            k_size = align(op.ksize * op.ksize * cin_pad, K_ALIGN)
            n_size = align(op.cout, COUT_ALIGN)
            ho, wo = op.out_shape[2], op.out_shape[3]
            launches.append(Launch(kind, variant, op.name, op.inputs[0], op.output, extra,
                                   stage=f"{kind} {h}x{w} {variant}",
                                   h=h, w=w, stride=op.stride, cin=cin, cin_pad=cin_pad, cin_stride=cs,
                                   k_size=k_size, cout=op.cout, n_size=n_size, ho=ho, wo=wo))
            shapes[op.output] = (ho, wo, n_size)
        elif op.kind in ("maxpool", "avgpool"):
            _, c, h, w = graph.shapes[op.inputs[0]]
            cs = storage_stride(c)
            launches.append(Launch("pool", "max" if op.kind == "maxpool" else "mean", op.kind, op.inputs[0],
                                   op.output, stage=f"pool {h}x{w}", h=h, w=w, cin=c, cin_pad=c, cin_stride=cs,
                                   ho=h // 2, wo=w // 2))
            shapes[op.output] = (h // 2, w // 2, cs)
        elif op.kind in ("relu", "add", "resize2x", "mul", "sigmoid", "flatten"):
            if op.kind == "relu" and op.output not in fused_relu:
                raise RuntimeError(f"unfused Relu {op.output}: its input {op.inputs[0]} is not a conv output")
            if op.kind == "add" and op.output not in fused_add:
                raise RuntimeError(f"unfused Add {op.output}")
            if op.kind == "mul":
                alias[op.output] = op.inputs[0]   # folded into the export
        else:
            raise NotImplementedError(op.kind)

    # the three fused heads, one per level, reading the tower's last conv output
    for level, (score, box, kps, _) in sorted(heads.items(), key=lambda kv: int(kv[0][1:])):
        src = score.inputs[0]
        _, cin, h, w = graph.shapes[src]
        cs = storage_stride(cin)
        cin_pad = align(cin, CIN_ALIGN)
        launches.append(Launch("conv3x3", "plain", level, src, f"head_{level}", stage=f"head {h}x{w}",
                               h=h, w=w, stride=1, cin=cin, cin_pad=cin_pad, cin_stride=cs,
                               k_size=align(9 * cin_pad, K_ALIGN), cout=30, n_size=64, ho=h, wo=w))
        shapes[f"head_{level}"] = (h, w, 64)

    # --- schedule: a launch runs once every tensor it reads exists --------------------
    for l in launches:
        if l.kind == "convert":
            continue                # its source is the raw input, which owns no buffer
        l.src = resolve(l.src)
        l.extra = resolve(l.extra) if l.extra else ""
    ordered: list[Launch] = []
    ready = {conv_in, graph.input}
    pending = list(launches[1:])
    ordered.append(launches[0])
    while pending:
        progressed = False
        for l in list(pending):
            needs = {l.src} | ({l.extra} if l.extra else set())
            if needs <= ready:
                ordered.append(l); pending.remove(l); ready.add(l.dst); progressed = True
                break                      # keep graph order as the tie-break
        if not progressed:
            raise RuntimeError("schedule stuck: " + ", ".join(f"{l.name}<-{l.src}/{l.extra}" for l in pending))

    # --- buffers by liveness ------------------------------------------------------------
    last_use: dict[str, int] = {}
    for i, l in enumerate(ordered):
        for t in (l.src, l.extra):
            if t:
                last_use[t] = i
    for level in heads:
        last_use[f"head_{level}"] = len(ordered)     # outputs live to the end
    def bytes_of(t: str, batch: int = 1) -> int:
        h, w, s = shapes[t]
        return h * w * s * 2 * batch
    buffers: list[int] = []            # size in bytes per image
    free: list[int] = []
    owner: dict[str, int] = {}
    for i, l in enumerate(ordered):
        need = bytes_of(l.dst)
        # a buffer large enough that is free, else a new one
        pick = None
        for b in sorted(free, key=lambda b: buffers[b]):
            if buffers[b] >= need:
                pick = b; break
        if pick is None:
            pick = len(buffers); buffers.append(need)
        else:
            free.remove(pick)
        owner[l.dst] = pick
        l.dst_buf = pick
        l.src_buf = owner[l.src] if l.src in owner else -1     # -1: the raw input
        l.extra_buf = owner[l.extra] if l.extra else -1
        for t in (l.src, l.extra):
            if t and t in owner and last_use.get(t) == i and t not in (f"head_{lv}" for lv in heads):
                free.append(owner[t])
    return ordered, shapes, buffers


def emit(ordered: list[Launch], shapes, buffers) -> None:
    heads = [l for l in ordered if l.name.startswith("h") and l.kind == "conv3x3" and l.cout == 30]
    kinds = {"convert": 0, "conv3x3": 1, "matmul": 2, "pool": 3}
    variants = {"": 0, "plain": 0, "relu": 1, "add": 2, "relu_add": 3, "add_resized": 4, "max": 5, "mean": 6}
    stems = sorted({l.stem for l in ordered})
    lines = ["// GENERATED by tools/gen_launch_table.py from the ONNX graph. Do not edit.",
             f"#define SCRFD_LAUNCH_COUNT {len(ordered)}",
             f"#define SCRFD_BUFFER_COUNT {len(buffers)}",
             f"#define SCRFD_KERNEL_COUNT {len(stems)}",
             "// bytes per image, per buffer",
             "static const size_t scrfd_buffer_bytes[SCRFD_BUFFER_COUNT] = {" + ", ".join(str(b) for b in buffers) + "};",
             "static const char *const scrfd_kernel_stems[SCRFD_KERNEL_COUNT] = {" + ", ".join(f'"{s}"' for s in stems) + "};",
             "// kind: 0 convert, 1 conv3x3, 2 matmul, 3 pool; variant: 0 plain, 1 relu, 2 add, 3 relu_add, 4 add_resized, 5 max, 6 mean",
             "static const scrfd_launch scrfd_launches[SCRFD_LAUNCH_COUNT] = {"]
    for l in ordered:
        lines.append(f'  {{{kinds[l.kind]}, {variants[l.variant]}, {stems.index(l.stem)}, "{l.name}", "{l.stage}", '
                     f'{l.h}, {l.w}, {l.stride}, {l.cin_pad}, {l.cin_stride}, {l.k_size}, {l.n_size}, {l.ho}, {l.wo}, '
                     f'{l.src_buf}, {l.dst_buf}, {l.extra_buf}}},')
    lines.append("};")
    head_lines = ", ".join(f'{{"{l.name}", {l.dst_buf}, {l.h}, {l.w}}}' for l in heads)
    lines.append(f"static const scrfd_head scrfd_heads[3] = {{{head_lines}}};")
    (ROOT / "host/graph_table.inc").write_text("\n".join(lines) + "\n")

    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build/launch_table.json").write_text(json.dumps(
        {"launches": [l.__dict__ | {"stem": l.stem} for l in ordered],
         "buffers": buffers, "shapes": shapes}, indent=1))

    # build lines: one per distinct stem
    build = ["# GENERATED by tools/gen_launch_table.py. Sourced by scripts/build_kernels.sh."]
    seen = set()
    for l in ordered:
        if l.stem in seen:
            continue
        seen.add(l.stem)
        if l.kind == "convert":
            continue   # hand-listed in build_kernels.sh
        if l.kind == "pool":
            build.append(f"compile pool2_f16 scrfd_pool2_f16 {l.stem} scrfd.pool2_f16.height={l.h} "
                         f"scrfd.pool2_f16.width={l.w} scrfd.pool2_f16.channels={l.cin_stride} "
                         f"scrfd.pool2_f16.take_max={1 if l.variant == 'max' else 0}")
        elif l.kind == "matmul":
            if l.variant == "add_resized":
                ns, src, sym = "scrfd.matmul_add_resized_f16_wmma", "matmul_add_resized_f16_wmma", "scrfd_matmul_add_resized_f16_wmma"
                build.append(f"compile {src} {sym} {l.stem} {ns}.k_size={l.k_size} {ns}.n_size={l.n_size} "
                             f"{ns}.height={l.ho} {ns}.width={l.wo}")
            else:
                ns, src, sym = "dinov3.matmul_bias_f16_wmma_af16_cf16", "matmul_bias_f16_wmma_af16_cf16", "dinov3_matmul_bias_f16_wmma_af16_cf16"
                build.append(f"compile {src} {sym} {l.stem} {ns}.k_size={l.k_size} {ns}.n_size={l.n_size}")
        else:
            suffix = "" if l.variant == "plain" else f"_{l.variant}"
            ns, src, sym = f"scrfd.conv3x3_f16_wmma{suffix}", f"conv3x3_f16_wmma{suffix}", f"scrfd_conv3x3_f16_wmma{suffix}"
            build.append(f"compile {src} {sym} {l.stem} {ns}.height={l.h} {ns}.width={l.w} {ns}.stride={l.stride} "
                         f"{ns}.cin_pad={l.cin_pad} {ns}.cin_stride={l.cin_stride} {ns}.k_size={l.k_size} {ns}.n_size={l.n_size}")
    (ROOT / "scripts/kernels.generated").write_text("\n".join(build) + "\n")


def main() -> None:
    graph = G.load()
    ordered, shapes, buffers = build_schedule(graph)
    emit(ordered, shapes, buffers)
    per_image = sum(buffers)
    stems = {l.stem for l in ordered}
    by_kind = {}
    for l in ordered:
        by_kind[f"{l.kind}/{l.variant}"] = by_kind.get(f"{l.kind}/{l.variant}", 0) + 1
    print(f"{len(ordered)} launches, {len(stems)} distinct kernels, {len(buffers)} buffers, "
          f"{per_image / 1e6:.1f} MB/image")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:<22s} {v}")


if __name__ == "__main__":
    main()
