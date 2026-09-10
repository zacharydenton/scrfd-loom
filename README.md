# scrfd-hrx

InsightFace det_10g (SCRFD-10GF) inference in Rust, using [hrx-rs](https://github.com/zacharydenton/hrx-rs)
for GPU execution and Loom compilation. One Cargo package provides the library
and CLI. Weights, kernels, activation buffers and I/O storage stay resident;
HRX graphs are recorded once per encountered batch size and replayed.

Requires Rust 1.91+, Linux x86-64 and a Radeon 8060S (`gfx1151`). HRX 0.4.0
provisions its verified runtime and compiler bundle; its native Linux bundle
requires glibc 2.43 or newer. Model weights are fetched through Hugging Face Hub when needed.

```bash
cargo build --release
```

## Library

`from_pretrained` loads the original `det_10g.onnx` from InsightFace's `buffalo_l` pack.
`onnx-protobuf` parses ONNX; the importer validates the supported graph, packs
weights and derives the fused launch schedule and buffer assignments.

```rust,no_run
use scrfd_hrx::{Scrfd, Options, Image, DetectionOptions};

# fn main() -> anyhow::Result<()> {
let mut model = Scrfd::from_pretrained(Options::default())?;
let bgr = vec![0u8; 640 * 640 * 3];
let faces = model.detect(
    Image { bgr: &bgr, width: 640, height: 640 },
    DetectionOptions::default(),
)?;
// Each Detection contains bbox [x1,y1,x2,y2], score and five [x,y] landmarks.
# Ok(())
# }
```

Images are packed uint8 BGR. The detector resizes to fit 640×640 with bilinear
interpolation, pads the right and bottom with black, and returns coordinates in
the original image. `detect_batch` chunks images at the resident batch limit.
`detect_letterboxed` accepts packed 640×640 BGR canvases, positive resize scales
and original `[width, height]` values when preprocessing is already done.

`Options` selects device index and resident batch (default 16, range 1–64).
Activation storage is 33.4 MB per resident image, plus input, weights and
terminal output storage. `DetectionOptions` defaults to score threshold 0.5, NMS
threshold 0.4 and capacity 4096 candidates per image; overflow returns an error.
`max_detections = 0` keeps all detections. Otherwise `Ranking::AreaAndCenter`
uses InsightFace's area/centre preference; `Ranking::Area` ranks by box area.
NMS uses inclusive coordinates. Empty batches return empty results.

The returned landmarks can be passed directly to
[arcface-hrx](https://github.com/zacharydenton/arcface-hrx) for alignment and
face embeddings.

## Model weights

`from_pretrained` and the CLI without `--model` use
[`immich-app/buffalo_l`](https://huggingface.co/immich-app/buffalo_l/tree/d09715916a0778919a770c343533641e250b8699),
file `detection/model.onnx`, pinned to revision `d09715916a0778919a770c343533641e250b8699`. The `hf-hub` 1.0 client reuses the shared
Hugging Face cache before downloading.
`HF_HOME` and `HF_HUB_CACHE` control its location; `HF_TOKEN` or a cached
Hugging Face login supplies authentication.

Use `Scrfd::load(path, options)` or `--model det_10g.onnx` for a local file.
An explicit local path never falls back to a download. `hub::weights(true)`
returns only cached weights. The CLI's `--offline` and `HF_HUB_OFFLINE=1`
also disable model downloads; `HRX_OFFLINE=1` separately disables runtime
bundle downloads.

The mirror contains the same original ONNX bytes used by the numerical tests.
InsightFace’s model terms still apply; the weights are not part of this package.

## CLI

```bash
cargo run --release -- --input image.png --output faces.json
```

The CLI reads PNG or JPEG images and writes detections as JSON. Add
`--benchmark 100` for timings. Image decoding is outside the timed calls.

## Execution and validation

Inference requires `&mut` access to the model. A model owns its stream; use
separate models for independent concurrent callers. GPU failures return errors
and make the session unusable. Drop releases owned resources through HRX.
The fixed production kernels are validated only for `gfx1151`.

Warm calls reuse compiled kernels, allocations and graphs. Inputs and terminal
outputs use coherent memory shared with the CPU; inference needs no GPU upload
or readback copies. Host access waits for completion. Graph dependencies track
buffer hazards, allowing independent branches to overlap while protecting
reused activations.
`benchmark` reports alternating graph/direct forward timings, excluding transfers;
the CLI also reports warm end-to-end timing. Both are synchronized host timings,
not hardware timestamp measurements. See [current measurements](docs/optimization-2026-09-10.md).

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
```

CPU tests run without a GPU or model files. Ignored tests fetch the pinned
weights when needed (`SCRFD_MODEL` overrides the path) and require hardware; they cover numerical agreement, changing inputs, partial batches and
graph replay. The unfused ONNX reference runs in float64 Rust. All nine head outputs
must exceed cosine 0.9999, with maximum error below 0.02 for scores and 0.15
for distances. Fixture detections require box IoU above 0.99, score error below
0.01 and landmark error below one pixel.

The lossless fixture preserves the pixels used for the original reference;
JPEG decoders can produce different pixels. These small fixtures establish
numerical agreement, not accuracy on other face datasets.

## License

Project code is Apache-2.0. Model weights have separate terms and are not
bundled with this crate. Downloads remain subject to those terms. See [third-party notices](THIRD_PARTY_NOTICES.md)
for model terms and retained source attribution.
