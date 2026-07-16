# Expand — Ascend C vs Torch vs PyPTO Comparison

## Operator
`Y[b,i,j] = X[b,i,0]` — broadcast expand last dim (1→384)

## Shapes
- X: `[B, 256, 1]`, FP16
- Y: `[B, 256, 384]`, FP16
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}

## Status: PARTIAL

## Ascend C Implementation: TRUE_DEVICE_IMPLEMENTATION
The kernel reads GM[1] per row → GetValue → Duplicate[384] → GM[384].
No host pre-expand. Source verified as genuine device-side compute.

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ⚠️ PARTIAL | B=1 verified; B=2..64 not persisted |
| Ascend C | ⚠️ UNVERIFIED | Kernel source confirmed; no correctness run |
| PyPTO | ✅ PASS | B=1..64 (per-row expand dispatch) |

## Performance (no msprof data)

| B | Torch | Ascend C | PyPTO |
|---|:-----:|:--------:|:-----:|
| Data | PENDING | PENDING | PENDING |

## Known Issues
1. No msprof profiling for any route (profiler INCOMPLETE)
2. Ascend C correctness not run (no run evidence)
3. PyPTO B=1 latency 29400 us is host dispatch overhead, not kernel compute
