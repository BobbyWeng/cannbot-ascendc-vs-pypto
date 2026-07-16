# Where Ascend C Root Cause Analysis

## Problem
Previous arithmetic approach (`cond*x1 + (1-cond)*x2`) produced all zeros due to Cast<uint8, half> precision issues and NaN propagation in arithmetic blending.

## Root Cause
1. `Cast<uint8, half>(condLocal, CAST_NONE)` on dav-2201 does not reliably produce exact 0.0/1.0 half values from uint8 0/1 inputs.
2. Arithmetic blending `cond*x1 + (1-cond)*x2` propagates NaN when either selected or unselected branch contains NaN — violating `torch.where` semantics where only the selected branch is evaluated.

## Fix
Use per-element scalar ternary select:
```
Cast(cond, uint8→half) → 0.0/1.0
for each element i:
  if cond[i] != 0.0:
    result[i] = x1[i]
  else:
    result[i] = x2[i]
```

This matches `torch.where` semantics exactly:
- Only the selected branch value is written to output
- Unselected branch NaN/Inf never propagates
- Signed zero follows the condition truth value

## Verification
- All 7 batches (B=1,2,4,8,16,32,64): **0 mismatches**
- Covers: all-true, all-false, alternating, random, sparse, dense, NaN branch, Inf branch, +0/-0
