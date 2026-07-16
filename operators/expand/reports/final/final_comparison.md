# Expand Final Comparison Report

## Status: COMPLETE

All three implementations (Torch, Ascend C, PyPTO) have passed full-batch correctness and profiling.

## Correctness

| Implementation | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---------------|-----|-----|-----|-----|------|------|------|
| Torch | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| Ascend C | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| PyPTO | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

All implementations produce bitwise-identical outputs (max_abs=0.0 on finite elements, NaN/Inf match).

## Ascend C Implementation

**Method**: Device-side Duplicate scalar per row
**Pipeline**: GM[1] → UB → GetValue(0) → Duplicate[384] → GM[384]
**Multi-core**: Rows distributed across 20 blocks

## Performance (torch.npu.Event, median_us, repeat=3)

| Batch | Torch | Ascend C | PyPTO | Notes |
|-------|-------|----------|-------|-------|
| 1 | 11.2 | 16.9 | 29.4ms* | *PyPTO includes per-row JIT dispatch (256 calls) |
| 2 | 12.8 | 17.0 | 58.2ms* | |
| 4 | 14.5 | 18.6 | 116.8ms* | |
| 8 | 17.3 | 28.5 | 233.9ms* | |
| 16 | 23.6 | 51.2 | 468.1ms* | |
| 32 | 36.1 | 92.5 | 937.6ms* | |
| 64 | 61.8 | 176.5 | 1.87s* | |

*PyPTO: per-row dispatch dominates (B×256 JIT kernel calls per logical Expand)

## Kernel Count

| Implementation | Kernel Type | Count per call |
|---------------|-------------|----------------|
| Torch | KERNEL_MIX_AIC | 1 |
| Ascend C | KERNEL_AIVEC | 1 |
| PyPTO | KERNEL_MIX_AIC + KERNEL_AICPU | B×256 × 3 |
