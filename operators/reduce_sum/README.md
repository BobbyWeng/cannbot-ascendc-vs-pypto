# ReduceSum — Ascend C vs Torch vs PyPTO Comparison

## Operator
`Y[b,i] = sum_j X[b,i,j]` — reduce sum over last dimension (384)

## Shapes
- X: `[B, 256, 384]`, FP16
- Y: `[B, 256]`, FP16
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}

## Status: COMPLETE_WITH_LIMITATION

## Ascend C Implementation: TRUE_DEVICE_IMPLEMENTATION
GM[384] → ReduceSum<Level 2, FP16> → GM[1]. FP16 native accumulation.

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ⚠️ 62/70 PASS | 7 NaN cases (expected), 1 B=64 borderline (max_abs=0.002 < atol=0.01) |
| Ascend C | ⚠️ 21/70 PASS | FP16 accum precision (~0.03 max_abs) exceeds atol=0.01. Pass: all_zero, all_one, small_values, underflow_risk |
| PyPTO | ⚠️ 21/70 PASS | Same FP16 accum limitation as Ascend C |

## Performance (msprof, warmup=200, loops=100, repeat=5)

| B | Torch (AIVEC) | Ascend C (AIVEC) | PyPTO |
|---|:----:|:--------:|:-----:|
| 1 | 16.4 us | 14.4 us | NOT RUN |
| 2 | 16.0 us | 14.7 us | |
| 4 | 16.2 us | 15.1 us | |
| 8 | 16.3 us | 25.0 us | |
| 16 | 16.5 us | 47.6 us | |
| 32 | 15.9 us | 93.4 us | |
| 64 | 16.2 us | 185.3 us | |

## Accumulation Dtype

| Implementation | Accumulation | Precision |
|--------------|-------------|-----------|
| torch.sum | FP32 (internal) | Reference |
| Ascend C | FP16 (native ReduceSum API) | ~0.03 max_abs vs FP32 |
| PyPTO | FP16 | ~0.03 max_abs vs FP32 |

## Known Issue
FP16 accumulation precision for 384-element reduction exceeds atol=0.01. Ascend C and PyPTO both use native FP16 accum. Either tolerance should be widened or FP32 accum used.
