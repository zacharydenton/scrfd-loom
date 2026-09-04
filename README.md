# scrfd-loom

insightface's `det_10g` face detector (SCRFD-10GF) written in **Loom**, AMD's kernel
language from [ROCm/hrx-system](https://github.com/ROCm/hrx-system), for the Radeon
8060S (gfx1151) in a Strix Halo APU. A sibling of
[dinov3-loom](https://github.com/zacharydenton/dinov3-loom), which built the toolchain
and the matmul this repo's convolution is derived from.

It is a drop-in for `insightface`'s `SCRFD.detect()`: same preprocessing, same anchor
decode, same NMS -- insightface's own code, vendored -- with only the network replaced.
On the sample image it returns the same six faces with IoU > 0.999, identical scores
and keypoints within 0.02 px of what insightface returns.

## Why

The deployed model is an ONNX file, and on this iGPU the fastest way to run it is
onnxruntime's MIGraphX provider: **5.07 ms per image**, after a 51-second graph
compile. That is 5.3 TFLOP/s effective on a 26.7-GFLOP forward pass, on a part whose
WMMA units sustain 16-24 TFLOP/s in dinov3-loom's matmuls. The graph is also exported
batch-1-only (`[1, 3, ?, ?]`), so MIGraphX cannot batch at all.

The question: is that the silicon, or the toolchain? Loom lets you write the
convolution directly, so the way to find out is to write it.

## Status

Complete, validated, benchmarked. One inference path, seven kernel sources, all of
them derived from or copied out of dinov3-loom's matmul except the two SIMT ones:

| kernel | what |
| --- | --- |
| `conv3x3_f16_wmma` (+ `relu`, `add`, `relu_add`) | 3x3 conv, pad 1, stride 1 or 2, as an implicit GEMM: the WMMA matmul with its A-staging load replaced by the im2col gather, so the `[M][K]` matrix never exists. 52 of the 58 convs, every residual/shortcut/PAFPN Add and every ReLU fused into the epilogue |
| `matmul_bias_f16_wmma_af16_cf16` (+ `add_resized`) | the 1x1 convs; `add_resized` is the FPN lateral with the 2x-upsampled coarser level added at `(y/2, x/2)` in the epilogue, so the Resize never exists |
| `pool2_f16` | MaxPool and AveragePool, 2x2 stride 2 |
| `hwc_u8_to_nhwc_f16` | the letterboxed BGR uint8 image to normalised RGB NHWC f16: `blobFromImage` and the relayout in one pass, so the host uploads bytes, never a blob |
| `im2col_f16` | the explicit gather -- the reference for the implicit one, and the permanent fallback |

The graph's 16 Adds, 2 Resizes, 36 ReLUs, the box-head Mul, the Sigmoid and all the
reshape plumbing are gone: **57 launches per forward pass**, from 158 ONNX nodes.
`tools/gen_launch_table.py` generates that schedule, the buffer assignment (five
buffers, 30 MB per image) and the kernel build list from the ONNX file; nothing about
the network is transcribed by hand.

Activations are NHWC f16, weights f16, accumulation f32 throughout.

## Correctness

`tools/reference.py` is a float64 NumPy interpreter of the ONNX graph, agreeing with
onnxruntime to 3e-6. Every kernel is graded against it, not against onnxruntime, so a
kernel bug cannot hide behind a matching bug in the harness. End to end
(`tools/validate.py`), all nine raw outputs vs onnxruntime on the sample image:

```
  PASS score/8     (12800, 1) cosine=0.9999990 max_abs=4.148e-04
  PASS box/8       (12800, 4) cosine=0.9999997 max_abs=1.085e-02
  PASS kps/8      (12800, 10) cosine=0.9999989 max_abs=8.899e-03
  ...
  PASS detections vs insightface: 6 faces (insightface 6); worst 1-IoU=0.0006 score delta=0.0001 kps=0.02 px
```

The detection check is against a fixture of insightface's *own* `SCRFD.detect()`
output, captured once with the real package, so the vendored decode is graded against
production rather than against itself.

## Benchmark

`tools/benchmark.py` times what a caller pays: `detect_batch` on real 1280x886 images,
letterbox to NMS, interleaved with onnxruntime+MIGraphX on the same box, best of three
rounds. MIGraphX's number is the network alone (`session.run` on a prepared blob; its
decode would come on top). Measured on a Radeon 8060S (gfx1151) with a 10-core CPU job
resident (load average 10 to 17), so treat the absolute numbers as a floor: on this box
the same batch-16 call measured anywhere from 2.83 to 3.53 ms/img from one run to the
next.

| configuration | img/s | ms/img | vs MIGraphX |
| --- | ---: | ---: | ---: |
| scrfd-loom `detect_batch`, batch 32 | **339.4** | 2.95 | **1.68x** |
| scrfd-loom `detect_batch`, batch 16 | 307.8 | 3.25 | 1.53x |
| scrfd-loom `detect_batch`, batch 8 | 285.0 | 3.51 | 1.41x |
| scrfd-loom `detect_batch`, batch 1 | 269.8 | 3.71 | 1.34x |
| onnxruntime + MIGraphX, network only, batch 1 | 201.6 | 4.96 | 1.00x |

Where a batch-16 call spends its time, best of seven, ms per image: letterbox 0.14,
the native call (upload, 57 launches, download, decode) 2.57, sort and NMS 0.05; end
to end 2.85. The network alone is 2.55 of that (`host/scrfd --repeat`), so the Python
API sits within 0.3 ms/img of the GPU. The deployed ONNX graph is `[1, 3, ?, ?]`:
MIGraphX cannot batch it, so its number is a latency. Detections are unchanged from
insightface's (worst 1-IoU 0.0006, score delta 1e-4, keypoints 0.02 px on the test
image).

## Replacing insightface

```python
# before
from insightface.model_zoo.scrfd import SCRFD
model = SCRFD("~/.insightface/models/buffalo_l/det_10g.onnx")
model.prepare(ctx_id=0, input_size=(640, 640))
det, kps = model.detect(image_bgr)

# after
from scrfd_loom import SCRFDLoom
model = SCRFDLoom()
det, kps = model.detect(image_bgr)          # same (n,5) boxes+scores, (n,5,2) keypoints
```

`detect_batch(images)` runs up to `max_batch` (default 16) images per GPU call, which
the deployed ONNX graph cannot do. The session is resident: kernels and weights load
once, buffers are sized for `max_batch` at construction, and each call is one ctypes
entry into `build/libscrfd.so`. Python does only the letterbox, straight into the
batch canvas, and NMS on the few dozen surviving rows; the normalisation, the network
and the anchor decode run in the native session, which returns thresholded candidates
rather than head tensors. `det_thresh` and `nms_thresh` default to insightface's 0.5
and 0.4; `max_candidates` (4096 per image) is the decode's capacity, and exceeding it
is an error rather than a silent truncation. The session is thread-safe, closable, and
a context manager, and paths can come from `SCRFD_LOOM_WEIGHTS`, `SCRFD_LOOM_KERNELS`
and `SCRFD_LOOM_LIBRARY`.

What to know before swapping: boxes agree with insightface to IoU > 0.999 and
keypoints to a fraction of a pixel, not bitwise; the kernels are compiled for 640x640
(insightface's default `det_size`) and gfx1151; the model is fetched from
`~/.insightface/models/buffalo_l/det_10g.onnx`, or `SCRFD_ONNX`.

## Running it

Needs the Loom toolchain from hrx-system (`scripts/env.sh` points at the build) and
ROCm for `hipcc`; `requirements.txt` for Python; the `det_10g.onnx` file from
insightface's `buffalo_l` pack.

```console
$ pip install -r requirements.txt
$ source scripts/env.sh
$ python3 tools/export_weights.py           # 52 padded f16 matrices, 10.9 MB
$ python3 tools/gen_launch_table.py         # schedule + kernel build list from the graph
$ ./scripts/build_kernels.sh                # 40 HSACOs, ~0.2 s
$ ./scripts/build_host.sh                   # host/scrfd CLI + build/libscrfd.so
$ ./scripts/test.sh                         # everything, ~3 minutes
```

`scripts/test.sh` is the one test command: formatting, the generated files against
their generators, the build, every unit test against float64, the runner's error paths,
then the end-to-end comparisons against onnxruntime and insightface.

## Layout

```
kernels/    the seven .loom sources and their generated variants
host/       scrfd.cpp + scrfd.h (resident session, decode, C ABI, CLI), graph_table.inc (generated)
tools/      graph.py (the ONNX graph, resolved), reference.py, decode.py (insightface's,
            vendored), export_weights.py, gen_launch_table.py, gen_variants.py, tests
scrfd_loom.py   the Python API
docs/       notes.md -- every lever tried, won or lost, with numbers
```

## Licence

Apache-2.0 (`LICENSE`). `tools/decode.py` is vendored from deepinsight/insightface, MIT.
