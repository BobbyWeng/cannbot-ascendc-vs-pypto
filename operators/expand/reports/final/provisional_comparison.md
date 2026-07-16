# Expand Provisional Comparison Report

**Status: PROVISIONAL** — Ascend C device-side kernel not yet implemented; full msprof profiling deferred.

## Correctness

| Implementation | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---------------|-----|-----|-----|-----|------|------|------|
| Torch | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| PyPTO | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| Ascend C | PASS* | PASS* | PASS* | PASS* | PASS* | PASS* | PASS* |

*Ascend C: host pre-expand + Add/Sub identity kernel. Not a device-side expand implementation.

## Profiler

Not yet completed. See profiler_matrix.csv for planned configuration.

## PyPTO Implementation Detail

- JIT kernel: 1D per-row `expand_clone([1] → [384])`
- Each batch dispatches B × 256 kernel calls
- Correctness verified: bitwise match on finite elements (max_abs_diff=0.0)

## Status

**INCOMPLETE** — Ascend C needs device-side kernel. Profiling deferred.
