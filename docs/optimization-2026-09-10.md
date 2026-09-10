# Inference optimization — 2026-09-10

Radeon 8060S (`gfx1151`), published HRX 0.4.0 and its pinned compiler.
Three paired runs reverse the executable order each round; each run uses
10 warmups and 300 samples. The table reports the median of the three run
medians for complete inference, including host preprocessing and I/O.
Model loading, compilation and image decoding are excluded. `max_batch=16`.

| Batch | Shipped median ms | Optimized median ms | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 3.145 | 2.755 | 12.4% |

Inputs are the existing lossless face fixture.
[Raw results](optimization-2026-09-10.json) include every run and its p95.
These are local measurements; background activity affects host latency.
One baseline run reached 3.683 ms and one optimized run reached 2.994 ms; the other optimized medians were 2.755 and 2.736 ms.
The [migration measurements](benchmark-2026-09-10.md) remain a historical baseline.

## Changes

Inputs and terminal outputs use HRX host-local, device-visible, coherent
allocations. The CPU copies input bytes into the resident input before graph
submission and reads output bytes only after completion. There are no GPU
upload or readback copies during inference. Weights and reused intermediate
activations remain device-local. Terminal outputs have separate allocations
so earlier layers never repeatedly read or write coherent output storage.
The session tracks pending work and rejects reuse after a GPU failure.

Graph edges now follow buffer hazards: reads wait for the previous writer;
a write also waits for all prior readers of that allocation. Tracking whole
allocations is conservative for sliced regions. This preserves activation-pool
reuse while allowing independent branches to overlap. Shape, binding and
constant validation are unchanged. The direct benchmark remains serial.

## Native primitives checked

The inspected `hrx-system` checkout was at `ecaaf7376f7d` with local compiler
edits. Measurements use the published bundle, not that modified compiler.
In `libhrx/src/libhrx/graph_exec.c`, graph launch flushes pending stream work;
eliminating separate transfers therefore removes submissions as well as copies.
Its barrier planner inserts barriers for dependencies, so unnecessary edges
also prevent overlap. The allocator's fine-grained host pools supply coherent
I/O; the Rust surface is `Stream::allocate_shared`.

Native `hrx_event_elapsed_time` currently subtracts host record timestamps
(`libhrx/src/libhrx/event.c`), so all reported timings explicitly use synchronized
host clocks. Stream events and semaphores support pipelining, but overlapping
requests would also require separate activation storage; that is outside this
synchronous API change.

The full numerical reference, fixture and changing-batch tests pass with the
same tolerances. The model arithmetic and HRX version are unchanged.
