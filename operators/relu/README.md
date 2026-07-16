# ReLU — Ascend C vs Torch vs PyPTO Comparison

## Operator
`y = max(x, 0)` — element-wise ReLU activation

## Shapes
- X / Y: `[B, 12, 256, 32]`, FP16
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}
- No broadcasting

## Status: COMPLETE ✅

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ✅ PASS | All 7 batches bitwise exact (signed-zero exempted) |
| Ascend C | ✅ PASS | All 7 batches bitwise exact |
| PyPTO | ✅ PASS | All 7 batches bitwise exact |

## Performance (primary compute kernel latency, µs)

| B | Torch (KERNEL_AIVEC) | Ascend C (KERNEL_AIVEC) | PyPTO (KERNEL_MIX_AIC) | PyPTO total (incl. AICPU) |
|---|:----:|:--------:|:------:|:--------:|
| 1 | 2.6 | 2.1 | 51.9 | 163.0 |
| 2 | 2.7 | 2.4 | 49.5 | 154.7 |
| 4 | 2.8 | 2.7 | 52.5 | 165.5 |
| 8 | 3.2 | 3.0 | 61.3 | 187.1 |
| 16 | 3.7 | 4.0 | 79.7 | 231.8 |
| 32 | 4.7 | 6.0 | 103.9 | 281.7 |
| 64 | 6.6 | 9.7 | 150.8 | 378.5 |

## Implementation Strategies

| Implementation | Kernel Type | Kernels/call | Notes |
|--------------|-------------|:------------:|-------|
| torch.relu | KERNEL_AIVEC | 1 | Single NPU kernel |
| Ascend C | KERNEL_AIVEC | 1 | Single kernel via <<<>>> |
| PyPTO | KERNEL_MIX_AIC + 2×AICPU | 3 | Compute + executor |

## Key Files
- `ascendc/src/relu_kernel.asc` — Ascend C kernel
- `pypto/src/relu_impl.py` — PyPTO implementation
- `reports/final/final_comparison.md` — Full comparison report

## Known Limitation
PyPTO produces 3 kernel events per logical call (1 KERNEL_MIX_AIC compute + 2 KERNEL_AICPU executor). The executor kernels are PyPTO runtime overhead, not compute. True compute kernel is KERNEL_MIX_AIC.

See `reports/final/final_comparison.md` for full analysis.
