# Transpose — Ascend C vs Torch vs PyPTO Comparison

## Operator
`Y[b,j,i] = X[b,i,j]` — permute [0,2,1], materialized contiguous output

## Shapes
- X: `[B, 256, 384]`, FP16
- Y: `[B, 384, 256]`, FP16
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}

## Status: PARTIAL

## Ascend C Implementation: TRUE_DEVICE_IMPLEMENTATION
The kernel does 16×16 tile-based transpose with element-wise swap.
No host pre-transpose. Source verified as genuine device-side compute.

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ⚠️ PARTIAL | B=1 only |
| Ascend C | ⚠️ UNVERIFIED | Kernel source confirmed; no correctness run |
| PyPTO | ⚠️ PARTIAL | Small shape PASS; large [256,384] BLOCKED_BACKEND |

## Performance (no msprof data)

| B | Torch | Ascend C | PyPTO |
|---|:-----:|:--------:|:-----:|
| Data | PENDING | PENDING | PENDING |

## Known Issues
1. No msprof profiling for any route (profiler INCOMPLETE)
2. Ascend C correctness not run
3. PyPTO large shape blocked at backend (CompileFunction)
