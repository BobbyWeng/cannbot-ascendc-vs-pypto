# Equal Ascend C Root Cause Analysis

## Problem
114 mismatches on B=1, concentrated in positions where `x1 == x2` (bit-identical FP16 values).
All output `0` where expected `1`.

## Root Cause
`Select` API with `VSEL_TENSOR_TENSOR_MODE` and `VSEL_TENSOR_SCALAR_MODE` on Atlas A2 (dav-2201) does not correctly expand the packed bitmask from `Compare`/`Compares` to per-element half 0/1 values. Despite documentation claiming mode 2 "progressively consumes mask bits", the actual hardware behavior on dav-2201 either:
1. Does not support the count-based Select API for mode 2/1 correctly, or
2. Uses a different mask consumption scheme than expected.

The `Compare` API produces correct packed bitmask output (each bit = one comparison result, 8 results per byte), but no vectorized Select path correctly expands these bits to byte-level results.

## Fix
Use `Compare` for vectorized packed mask production (fast, hardware-accelerated), then **scalar expansion** to convert packed bits to individual byte results:

```
for each 128-element chunk:
  Compare(x1, x2, EQ, 128) → 16 bytes of packed mask
  for each byte j (0..15):
    for each bit b (0..7):
      output[j*8 + b] = (byte[j] >> b) & 1
```

This is correct but slower than a hypothetical vectorized Select path.

## Verification
- All 7 batches (B=1,2,4,8,16,32,64): **0 mismatches**
- Edge cases (NaN, Inf, +0/-0, adjacent values): all match `torch.eq`
