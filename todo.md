# Task: Optimize Where, Transpose, Equal — Ascend C Performance

## Results Summary

| Operator | Before | After | Torch | Speedup | Correctness |
|----------|--------|-------|-------|---------|-------------|
| Where    | 248us  | 7.6us | 11.6us| 32.6x   | PASS ✅     |
| Transpose| 85us   | 85us* | 14.1us| 1.0x    | N/A        |
| Equal    | 50us   | 41.3us| 11.5us| 1.2x    | PASS ✅     |

Transpose: No improvement found — DMA-bound. Original kernel was already using optimal tile size (32x32 for B=1). The bottleneck is strided DMA access pattern inherent to transpose. Attempts with Gather/merged-approach caused hangs.

## Where — COMPLETE ✅
- Approach: Compare+Select API for fully vectorized select
- 7.6us B=1 (32.6x vs 248us)
- PASS correctness (bitwise)
- Source: where_kernel.asc (Compare + Select with SELMODE::VSEL_TENSOR_TENSOR_MODE)

## Equal — COMPLETE ✅
- Approach: Compare + scalar bit expansion (vectorized Compare, minimal scalar expansion)
- 41.3us B=1 (1.2x vs 50us)
- PASS correctness (bitwise)
- Select API not functional on this arch for the half→uint8 conversion path
