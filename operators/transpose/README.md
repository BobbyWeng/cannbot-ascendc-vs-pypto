# Transpose — Ascend C vs Torch vs PyPTO Comparison

## Operator
`Y[b,j,i] = X[b,i,j]` — permute [0,2,1], materialized contiguous output

## Shapes
- X: `[B, 256, 384]`, FP16
- Y: `[B, 384, 256]`, FP16 (materialized, not a view)
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}

## Status: INCOMPLETE

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ⏳ PENDING | |
| Ascend C | ⏳ PENDING | |
| PyPTO | ⏳ PENDING | |

## Performance (primary compute kernel latency, µs)

| B | Torch (KERNEL_AIVEC) | Ascend C (KERNEL_AIVEC) | PyPTO (KERNEL_MIX_AIC) | PyPTO total (incl. AICPU) |
|---|:----:|:--------:|:------:|:--------:|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |
| 16 | — | — | — | — |
| 32 | — | — | — | — |
| 64 | — | — | — | — |

## Implementation Strategies

| Implementation | Kernel Type | Kernels/call | Notes |
|--------------|-------------|:------------:|-------|
| torch.transpose | KERNEL_AIVEC | 1 | Single NPU kernel |
| Ascend C | KERNEL_AIVEC | 1 | Single kernel via <<<>>> |
| PyPTO | KERNEL_MIX_AIC + 2×AICPU | 3 | Compute + executor |

## Key Files
- `ascendc/src/transpose_kernel.asc` — Ascend C kernel
- `pypto/src/transpose_impl.py` — PyPTO implementation
- `reports/final/final_comparison.md` — Full comparison report

## Known Limitation
PyPTO produces 3 kernel events per logical call (1 KERNEL_MIX_AIC compute + 2 KERNEL_AICPU executor). The executor kernels are PyPTO runtime overhead, not compute. True compute kernel is KERNEL_MIX_AIC.

See `reports/final/final_comparison.md` for full analysis.
