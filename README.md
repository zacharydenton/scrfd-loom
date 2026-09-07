# scrfd-loom

InsightFace's `det_10g` face detector (SCRFD-10GF) implemented in
[Loom](https://github.com/ROCm/hrx-system) for the AMD Radeon 8060S (gfx1151).
It returns face boxes, scores and five landmarks from BGR images.

The network uses f16 weights and activations with f32 accumulation. Preprocessing,
anchor decoding and non-maximum suppression follow InsightFace. Weights, kernels
and GPU buffers stay resident between calls. Combine it with
[arcface-loom](https://github.com/zacharydenton/arcface-loom) for face embeddings.

## Performance and validation

Recorded on a Radeon 8060S, best of three interleaved rounds with other CPU work
running. SCRFD-Loom includes letterboxing, inference, decode and NMS on 1280×886
images. **The MIGraphX baseline measures the network only**, on a prepared blob.

| Runtime | Batch | Images/s | ms/image |
| --- | ---: | ---: | ---: |
| scrfd-loom `detect_batch` | 1 | 334.4 | 2.99 |
| scrfd-loom `detect_batch` | 8 | 400.6 | 2.50 |
| scrfd-loom `detect_batch` | 16 | 410.9 | 2.43 |
| scrfd-loom `detect_batch` | 32 | 422.6 | 2.37 |
| ONNX Runtime + MIGraphX, network only | 1 | 201.9 | 4.95 |

The supplied ONNX graph has a fixed batch dimension of 1. Loom supports batches
up to 64. See [`tools/benchmark.py`](tools/benchmark.py) for the timing procedure.

On InsightFace's sample image, validation finds the same six faces as its
reference fixture: worst box IoU error (`1 − IoU`) **0.0006**, score difference
**0.0001**, and landmark difference **0.02 pixels**. All nine network outputs
are also compared with ONNX Runtime. Individual kernels use a float64 NumPy
reference. These are regression checks on a small fixture, not a detection
accuracy evaluation; validate representative images before changing a pipeline.

## Build

Requires Linux x86-64, a gfx1151 GPU, ROCm, Python 3.11+, the Loom compiler, and
`det_10g.onnx` from InsightFace's `buffalo_l` pack.

Follow the [build guide](docs/building.md) to build the pinned public compiler
revision and obtain and verify the model. Then, from this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
source scripts/env.sh
python tools/export_weights.py
python tools/gen_launch_table.py
./scripts/build_kernels.sh
./scripts/build_host.sh
./scripts/test.sh
```

`test.sh` rebuilds the assets and runs the full suite. `--quick` skips the final
ONNX Runtime, fixture and Python API comparisons; it still needs the GPU and
model. Standard CPU ONNX Runtime suffices for validation. The benchmark needs
an ONNX Runtime build with MIGraphX.

The wheel contains the Python loader and decode module. Outside a checkout,
set `SCRFD_LOOM_WEIGHTS`, `SCRFD_LOOM_KERNELS` and `SCRFD_LOOM_LIBRARY` to the
exported weights, compiled kernels and `libscrfd.so`. Build and runtime overrides
are in the [build guide](docs/building.md#paths-and-runtime-selection).

## Python API

```python
from scrfd_loom import SCRFDLoom

with SCRFDLoom(max_batch=16) as model:
    model.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)
    boxes, landmarks = model.detect(image_bgr)
    results = model.detect_batch(images_bgr)
```

`boxes` has shape `(N, 5)`: `x1, y1, x2, y2, score`. `landmarks` has shape
`(N, 5, 2)`. Coordinates refer to the original image. `detect_batch` returns one
`(boxes, landmarks)` pair per image, processing up to `max_batch` at a time.

`prepare()` is optional. It accepts `ctx_id=0`, `input_size=(640, 640)`,
`det_thresh` and `nms_thresh`; CPU fallback and other input sizes or device IDs
are rejected. Defaults are 0.5 for detection and 0.4 for NMS. The candidate
capacity is 4096 per image; overflow raises an error. Increase `max_candidates`
at construction if needed.

`detect(image, input_size=None, max_num=0, metric="default")` follows
InsightFace's area/centre ranking when limiting detections; `metric="max"`
ranks by area. `input_size` may be `None` or `(640, 640)`.

For an existing `FaceAnalysis` pipeline, replace both detector references:

```python
with SCRFDLoom() as model:
    app.models["detection"] = app.det_model = model
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(image_bgr)
```

If your pipeline already letterboxes images, use
`detect_letterboxed(canvases, det_scales)`. Canvases must be `(B, 640, 640, 3)`
uint8 BGR, with content aligned to the top left. Supply original `(height, width)`
values in `image_shapes` when using the default `max_num` ranking with scales.
Keep preprocessing consistent when comparing results between implementations.

Calls on one model are serialized and thread-safe. Use `close()` or a context
manager to release GPU resources, and create sessions after forking. The
ONNX-specific `session`, `model_file` and raw `forward` interfaces are not provided.
Other GPUs are unvalidated.

## Implementation

The ONNX graph generates the launch schedule and five activation buffers
(33.4 MB per image). Fused convolutions, FPN additions, pools and input conversion
reduce the graph to 57 launches. The native host decodes candidate detections;
Python performs letterboxing and NMS.

- [`kernels/`](kernels/): Loom convolution, conversion and pooling kernels.
- [`host/`](host/): resident C ABI, detector CLI and kernel test runner.
- [`tools/`](tools/): graph parsing, export, generation, reference and tests.
- [Engineering notes](docs/notes.md): implementation details and measurements.

## License

Project code is [Apache-2.0](LICENSE). InsightFace-derived preprocessing and
postprocessing use MIT; see [third-party notices](THIRD_PARTY_NOTICES.md).

Model weights have separate terms. InsightFace distributes `buffalo_l` for
non-commercial research; other uses require appropriate model licensing.
Neither the ONNX model nor exported weights are included. See
[InsightFace's policy](https://github.com/deepinsight/insightface#license).
