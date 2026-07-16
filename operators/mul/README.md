# Mul — Ascend C vs PyPTO Comparison

## Operator
Element-wise multiplication: `Y = X1 * X2`

## Status
- **Correctness**: ✅ PASS (strict bitwise for all 7 batch sizes)
- **Benchmarks**: ✅ Complete (msprof profiler measurement)

## Results Summary

| Implementation | Kernel Type | Kernels/call | Device kernel (B=1) | Device kernel (B=64) |
|--------------|-------------|-------------|--------------------|--------------------|
| torch.mul | KERNEL_AIVEC | 1 | 3.9 us | 12.5 us |
| Ascend C | KERNEL_AIVEC | 1 | 3.1 us | 17.6 us |
| PyPTO | KERNEL_MIX_AIC + 2×AICPU | 3 | 177.4 us | 496.1 us |

See `reports/final/final_comparison.md` for full analysis.
