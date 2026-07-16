# Add (4-Input) — Ascend C vs Torch vs PyPTO Comparison

## Operator
`Y = ((X1 + X2) + X3) + X4` — four-input element-wise addition

## Shapes
- X1, X2, X3, X4 / Y: `[B, 256, 384]`, FP16
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}
- No broadcasting

## Status: COMPLETE ✅

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ✅ PASS | All 7 batches bitwise exact |
| Ascend C | ✅ PASS | All 7 batches bitwise exact |
| PyPTO | ✅ PASS | All 7 batches bitwise exact |

## Performance (median kernel latency, µs)

| B | Torch | Ascend C | PyPTO |
|---|:-----:|:--------:|:-----:|
| 1 | 37.5 | 6.6 | 397.3 |
| 2 | 39.3 | 6.6 | 392.5 |
| 4 | 36.0 | 6.6 | 389.1 |
| 8 | 36.7 | 8.0 | 386.7 |
| 16 | 37.1 | 8.0 | 390.8 |
| 32 | 38.0 | 12.6 | 391.1 |
| 64 | 39.9 | 22.3 | 385.7 |

## Implementation Strategies

| Implementation | Kernel Type | Kernels/call | Notes |
|--------------|-------------|:------------:|-------|
| torch.add (3×) | KERNEL_AIVEC | 3 | Three sequential torch.add calls |
| Ascend C (fused) | KERNEL_AIVEC | 1 | Single fused 4-input add |
| PyPTO (3× chained) | KERNEL_MIX_AIC + 2×AICPU | 3 × 3 = 9 | Three binary add calls, each with 3 kernel events |

## Key Files
- `ascendc/src/add_kernel.asc` — Ascend C kernel
- `pypto/src/add_impl.py` — PyPTO implementation
- `reports/final/final_comparison.md` — Full comparison report

See `reports/final/final_comparison.md` for full analysis.
