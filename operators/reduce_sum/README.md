# ReduceSum — Ascend C vs Torch vs PyPTO Comparison

## Operator
`Y[b,i] = sum_j X[b,i,j]` — reduce sum over last dimension (384)

## Shapes
- X: `[B, 256, 384]`, FP16
- Y: `[B, 256]`, FP16
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}

## Status: PARTIAL

## Ascend C Implementation: TRUE_DEVICE_IMPLEMENTATION
The kernel uses Level 2 ReduceSum API: GM[384] → ReduceSum → GM[1].
No host pre-reduce. Source verified as genuine device-side compute.

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ⚠️ PARTIAL | B=1 random_finite only |
| Ascend C | ⚠️ UNVERIFIED | Kernel source confirmed; no correctness run |
| PyPTO | ⚠️ UNVERIFIED | Orchestrator SUCCESS; no results saved |

## Performance (no msprof data)

| B | Torch | Ascend C | PyPTO |
|---|:-----:|:--------:|:-----:|
| Data | PENDING | PENDING | PENDING |

## Accumulation Dtype

| Implementation | Accumulation |
|--------------|-------------|
| torch.sum | FP32 (internal) |
| Ascend C | FP16 (native ReduceSum API for half) |
| PyPTO | FP16 |

## Known Issues
1. No msprof profiling for any route (profiler INCOMPLETE)
2. Ascend C and PyPTO correctness not run/persisted
3. FP16 accumulation precision differences expected vs FP32 torch
