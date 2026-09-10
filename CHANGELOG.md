# Changelog

## 0.1.0 — Rust migration

- Renamed `scrfd-loom` to `scrfd-hrx`.
- Replaced the Python package and C ABI with a Rust library and CLI using HRX 0.4.0.
- Load original model files directly; weight conversion and graph scheduling run in Rust.
- Cache resident GPU graphs by batch size and reuse activation and readback allocations.
- Removed standalone HIP hosts, build scripts, Python tooling and obsolete experiments.

This replaces the former APIs; there are no deprecated compatibility wrappers.
