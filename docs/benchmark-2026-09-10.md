# HRX migration measurements — 2026-09-10

Radeon 8060S (`gfx1151`), HRX 0.4.0, batch 1. Each measurement uses
10 warmups and 100 samples. Forward measurements alternate graph replay and
direct dispatch and synchronize each sample. End-to-end measurements include
preprocessing, upload, inference and download; image decoding and model setup
are excluded. These are local measurements, not cross-machine performance claims.

| Path | Median ms |
| --- | ---: |
| HRX graph forward | 2.325 |
| HRX direct forward | 2.343 |
| Rust end-to-end | 3.091 |
| Former Python API end-to-end, old compiler | 2.868 |

Full distributions and setup timings: [JSON](benchmark-2026-09-10.json).
Earlier benchmark files describe the former implementation and toolchain.

The Rust end-to-end path remains approximately 9% slower than the old API on
this image. Reusable queued readback, SIMD resizing, resident canvas storage
and logit filtering reduced the initial regression. Graph replay is faster than
direct HRX dispatch; the remaining difference is in preprocessing/transfers and
runtime overhead, not an improvement claimed for the migration.
