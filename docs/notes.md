# Notes: SCRFD-10GF (`det_10g`) in Loom on gfx1151

Sibling of `~/code/dinov3-loom`; its `docs/notes.md` holds the Loom rules this repo
inherits (Lever 5: never `index.assume` a fragment-load index; Lever 7: 3
workgroups/CU is the measured optimum; Lever 9: classify stages by what bounds them;
never `where` on a launch argument). This file records what is specific to a
convolutional detector, and every lever tried here -- won or lost.

## The graph

`~/.insightface/models/buffalo_l/det_10g.onnx`, opset 11, 158 nodes, 4.2 M params,
BatchNorm folded. **58 convs, all `group=1`**: 52 are 3x3 (46 stride 1, 6 stride 2,
pad 1), 6 are 1x1. ReLU x36, Sigmoid x3. 16 Adds; 1 MaxPool k2s2; 3 AveragePool k2s2
(downsample shortcuts, each into a 1x1); 2 nearest-2x Resizes; a scalar `Mul` after
each box head conv (0.846, 0.900, ... -- a learned per-level scale). No depthwise, no
GroupNorm. `tools/graph.py` is the single source of truth: it folds the dynamic-shape
plumbing at 640 and every other tool walks its op list.

At 640x640: **26.7 GFLOP/img**, 2.2x a DINOv3 image. 320^2 17%, 160^2 33%, 80^2 36%,
40^2 7%, 20^2 8%. Channels 3/28/56/80/88/224; head outputs 2/8/20 per level.

## The oracle

`tools/reference.py` is a float64 NCHW interpreter of the graph (explicit im2col +
matmul). Against onnxruntime CPU f32 on all nine raw outputs: **max rel-to-max error
5e-7 to 3e-6.** 5.8 s per image; that is the price of being the ground truth.

`tools/decode.py` vendors insightface's letterbox, `blobFromImage`, anchor centres,
`distance2bbox`/`distance2kps` and NMS. Graded against insightface's *own*
`SCRFD.detect()` on `t1.jpg` (fixture captured once with the real package):
**6 faces, IoU 1.0000 on every box, identical scores, 0.000 px on every keypoint.**

## Baseline

ORT + MIGraphX on this box: **5.07 ms/img at batch 1** (51 s graph compile), which is
5.3 TFLOP/s effective. ORT CPU: 67 ms. The deployed graph is `[1, 3, ?, ?]` --
batch-1-only -- so MIGraphX has no batched number to compete with.

## Export

`tools/export_weights.py`: `[Cout_pad][K_pad]` f16 in the gather order
`k = (dy*3+dx)*Cin_pad + c`; Cin padded to 4 (only the stem's 3 needs it), K to 32,
Cout to 64 so a 32-wide N tile can read the same file. The three head convs per
level are fused into one of Cout 30 -> 64 in the order `[score | box*scale | kps]`,
which is insightface's anchor-major row order once NHWC is reshaped. 52 matrices,
10.8 MB f16. FLOP-weighted padding overhead 1.4%.

## Kernels

- `nchw_to_nhwc_f16`: the blob -> NHWC f16 with a zero 4th channel. Exact; 51 us for
  two 640^2 images, 191 GB/s -- bandwidth-bound, as it should be.
- `im2col_f16`: the explicit gather, and the reference for the implicit one --
  identical address arithmetic. Exact against NumPy at a tiny shape covering every
  halo case and at 640^2/160^2/80^2, stride 1 and 2. First prover rejection of this
  repo: a `vector<4xf16>` at channel `c = k % Cin_pad` needs `c + 4 <= Cin_pad`, and
  `rem` alone does not tell the subrange prover that `c` is a multiple of 4 -- it
  passed at Cin_pad 4 only because `c` folds to 0 there. Stating `[le(c, Cin_pad-4),
  mul(c, 4)]` on the load path (legal: no fragment op downstream) fixes it.
- **Day-one conv** = im2col + the af16_cf16 WMMA matmul, real weights, vs float64:
  cosine 0.99999996 on c00/c03/c10/c27. The matmul runs 5.7-11.9 TFLOP/s on padded
  K, and materialising the columns costs 30-55% of the matmul's time on top. That
  overhead is what the implicit-GEMM conv removes; this path stays as the fallback.
