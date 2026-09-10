# Third-party notices

## InsightFace source code

`src/detection.rs` contains preprocessing and
post-processing code derived from the InsightFace project:

- Project: <https://github.com/deepinsight/insightface>
- Authors: Jiankang Deng and Jia Guo
- License: MIT

MIT License

Copyright (c) 2018 Jiankang Deng and Jia Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## InsightFace model weights

The `buffalo_l` / `det_10g.onnx` model is not distributed by this repository
and is not covered by this project's Apache-2.0 license. InsightFace currently
describes its supplied pretrained models as available for non-commercial
research use unless separate authorization is obtained. Users are responsible
for reviewing the model terms at <https://github.com/deepinsight/insightface>
and obtaining any permission their use requires.

## Validation fixture

`tests/fixtures/t1.png` preserves the decoded pixels of InsightFace's
[`t1.jpg`](https://github.com/deepinsight/insightface/blob/7fadd420c2351d0ffa8cac403421c1a3ed733365/python-package/insightface/data/images/t1.jpg).
The accompanying JSON records the existing InsightFace reference results.
These retain the InsightFace MIT attribution above. No model weights are included.

## Hugging Face download source

The optional automatic download uses [`immich-app/buffalo_l`](https://huggingface.co/immich-app/buffalo_l/tree/d09715916a0778919a770c343533641e250b8699),
revision `d09715916a0778919a770c343533641e250b8699`, file `detection/model.onnx`.
The validated file SHA-256 is `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`.
These are the original model bytes; hosting them on Hugging Face does not change
the model terms above. Weights are cached outside the Cargo package.
