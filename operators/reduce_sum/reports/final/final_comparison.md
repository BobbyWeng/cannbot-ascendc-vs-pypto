# ReduceSum Final Comparison Report

## Status: COMPLETE

All three implementations (Torch, Ascend C, PyPTO) have passed full-batch correctness and profiling.

## Correctness

| Implementation | B=1..64 Core Cases | Notes |
|---------------|-------------------|-------|
| Torch | PASS | FP32 accumulation; all cases PASS |
| Ascend C | PASS | FP16 accumulation; max_abs=0.031 vs torch (expected FP16 precision diff) |
| PyPTO | PASS | Same FP16 accumulation behavior as Ascend C |

Core cases: all_zero, all_one, small_values, nan, inf — all bitwise exact.

## Ascend C Implementation

**Method**: Level 2 ReduceSum per row
**Pipeline**: GM[384] → UB → ReduceSum<half> → GM[1]
**Multi-core**: Rows distributed across 20 blocks
**Accumulation**: FP16 (native half ReduceSum)

## Performance (torch.npu.Event, median_us, case=random_finite, repeat=3)

| Batch | Torch | Ascend C | PyPTO |
|-------|-------|----------|-------|
| 1 | 27.8 | 20.8 | 118.2 |
| 2 | 29.5 | 19.8 | 119.5 |
| 4 | 31.2 | 31.0 | 117.8 |
| 8 | 35.6 | 31.2 | 120.3 |
| 16 | 44.1 | 54.0 | 125.6 |
| 32 | 60.2 | 100.1 | 132.4 |
| 64 | 95.7 | 191.9 | 148.1 |

## Precision Comparison

| Metric | Torch | Ascend C | PyPTO |
|--------|-------|----------|-------|
| Accumulation dtype | FP32 | FP16 | FP16 |
| Max abs diff vs FP32 ref | 0.0 | 0.031 | 0.031 |
| Max rel diff vs FP32 ref | 0.0 | 0.001 | 0.001 |

## Kernel Count

| Implementation | Kernel Type | Count per call |
|---------------|-------------|----------------|
| Torch | KERNEL_MIX_AIC | 1 |
| Ascend C | KERNEL_AIVEC | 1 |
| PyPTO | KERNEL_MIX_AIC + KERNEL_AICPU | ~3 |
