# Limitation Classification Matrix

## Hardware Capability Limitations

| Operator | Route | Limitation | Evidence | Classification |
|----------|-------|------------|----------|----------------|
| where | Ascend C | Scalar ternary loop 37x slower than vectorized | Cast<uint8,half> reliability, NaN propagation from arithmetic blending | ASCENDC_COMPILER_LIMITATION (not strictly HW) |
| expand | Ascend C | Need on-device broadcast Duplicate API | Current kernel does CPU precompute | NOT_HARDWARE — implementation bug |
| transpose | Ascend C | Need on-device tile-based transpose | Current kernel does CPU precompute | NOT_HARDWARE — implementation bug |
| reduce_sum | Ascend C | Need full FP32 reduction on-device | Current kernel does CPU pre-reduce | NOT_HARDWARE — implementation bug |

**No confirmed HARDWARE_CAPABILITY_LIMITATION** exists for any operator. All blocker cases are:
- PYPTO_BACKEND_LIMITATION (div, equal, where)
- ASCENDC_COMPILER_LIMITATION (where scalar fallback)
- IMPLEMENTATION_BUG (expand, transpose, reduce_sum CPU precompute)
- MEASUREMENT_LIMITATION (not, or flat PyPTO latency)

## False "Hardware Limitation" Claims Detected

None of the current reports claim "hardware limitation" for blocked cases. All correctly classify as backend/framework issues. This is correct behavior.

## Framework/Backend Limitations

| Operator | Route | Detailed Issue | Min Case Working | Max Case Failing |
|----------|-------|----------------|------------------|-------------------|
| div | PyPTO | Broadcast CompileFunction | [1,32]/[1,1] | [3072,256]/[3072,1] |
| equal | PyPTO | Compare result lowering broken | None | Even [1,32] FP16->FP16 wrong |
| where | PyPTO | Expand pass dtype mismatch | None | Same-shape fails |

## Implementation Bugs

| Operator | Route | Issue | Fix Priority |
|----------|-------|-------|--------------|
| or | PyPTO | bitwise_or instead of logical_or | P1 - semantic correctness |
| not | PyPTO | Flat latency unverified | P1 - needs msprof verification |
| expand | Ascend C | CPU precompute for broadcast | P1 - needs device-side fix |
| transpose | Ascend C | CPU precompute for permutation | P1 - needs device-side fix |
| reduce_sum | Ascend C | CPU pre-reduction for reduce | P1 - needs device-side fix |
