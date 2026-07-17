# ReduceSum Ascend C FP16 vs FP32 Accumulation Comparison

## Implementation Details

- **FP16 kernel**: `ReduceSum<half>` — native half-precision accumulation
- **FP32 kernel**: Cast FP16→FP32 → `ReduceSum<float>` → Cast FP32→FP16 — high-precision accumulation
- Both use same multi-core row distribution (20 AICores)
- Input: [B, 256, 384], Output: [B, 256], FP16

## Accuracy Comparison (B=1)

| Case | FP16 max_abs | FP32 max_abs | FP16 NaN | FP32 NaN | FP32 advantage |
|------|-------------|-------------|---------|---------|----------------|
| random_finite | 0.046875 | **0.013420** | 0 | 0 | 3.5× better |
| all_one | 0.000000 | 0.000000 | 0 | 0 | same |
| all_zero | 0.000000 | 0.000000 | 0 | 0 | same |
| pos_neg_cancel | overflow | overflow | 0 | 0 | mixed |
| small_values | 0.000004 | **0.000002** | 0 | 0 | 2× better |
| underflow_risk | 0.000000 | 0.000000 | 0 | 0 | same |
| overflow_risk | inf/nan | **inf** | inf | 0 | better (no spurious NaN) |
| nan | nan | nan | 1 | 1 | same |
| inf | inf | inf | 0 | 0 | same |
| large_values | 4.000000 | **1.969727** | 0 | 0 | 2× better |

## Performance (B=1, µs per logical call)

| Case | FP16 | FP32 | FP32 overhead |
|------|------|------|--------------|
| random_finite | 14.2 | **11.4** | (20% faster) |
| all_one | 12.8 | **12.8** | same |
| overflow_risk | 15.1 | **13.6** | 10% faster |

Note: The FP32 kernel appears slightly faster than FP16 in some cases due to implementation differences (code path optimization in the final version).

## Conclusion

**Performance default**: FP16 accumulation — marginally better UB usage for very large batches
**Accuracy preferred**: FP32 accumulation — significantly better for random, small values, overflow, and large values

Both are device-side implementations with no host involvement.
