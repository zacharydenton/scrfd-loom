# Changelog

## Unreleased

- Fetch pinned pretrained weights through the shared Hugging Face cache by default; retain local-file and offline loading.
- Require Rust 1.91 for the HF Hub 1.0 dependency stack.

- Use resident coherent inputs and terminal outputs to remove GPU transfer submissions.
- Record graph dependencies from buffer hazards so independent branches can overlap.

## 0.1.0 — Rust migration

- Renamed `scrfd-loom` to `scrfd-hrx`.
- Replaced the Python package and C ABI with a Rust library and CLI using HRX 0.4.0.
- Load original model files directly; weight conversion and graph scheduling run in Rust.
- Cache resident GPU graphs by batch size and reuse activation and readback allocations.
- Removed standalone HIP hosts, build scripts, Python tooling and obsolete experiments.

This replaces the former APIs; there are no deprecated compatibility wrappers.
