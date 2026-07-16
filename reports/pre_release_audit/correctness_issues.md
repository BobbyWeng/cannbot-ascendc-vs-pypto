# Correctness Issues Found (Pre-Release Audit)

## P0 — Must Fix Before Release

### 1. Not Ascend C — ALL batches FAIL (script bug)
- **Evidence**: `operators/not/reports/correctness/ascendc_correctness.json` shows all 7 batches FAIL
- **Root cause**: correctness.py looks for `reference_b*_bool.bin` but files are named `reference_b*_{case}.bin`
- **Impact**: Ascend C Not correctness is entirely unverified
- **Reports falsely claim**: PASS

### 2. Or Ascend C — ALL batches FAIL (same script bug)
- **Evidence**: `operators/or/reports/correctness/ascendc_correctness.json` shows all 7 batches FAIL
- **Same root cause** as Not
- **Reports falsely claim**: PASS

### 3. Or PyPTO — bitwise_or instead of logical_or
- **Evidence**: `operators/or/pypto/src/or_impl.py` uses `pypto.bitwise_or` not `pypto.op.logical_or`
- **Impact**: For non-0/1 uint8 inputs, results differ from spec
- **Only passes** because test data is strictly 0/1

## P1 — Should Fix

### 4. Div torch correctness coverage gap
- B=4,8,16,32 were SKIP in stored `correctness_results.json` (ref files didn't exist at time)
- Reference files now exist; needs re-run

### 5. Expand/Transpose/ReduceSum — torch correctness only B=1
- All three show `all_pass: true` but only B=1 data exists
- B=2..64 missing across all three

### 6. ReduceSum — fp32_accum reference fails on NaN case even at B=1
- Documented but needs investigation

### 7. Expand/Transpose/ReduceSum — Ascend C correctness never run
- Host precompute kernels exist but Ascend C correctness.py never executed
- Output BINs exist but no correctness JSON

## P2 — Can Do Later

### 8. ULP measurement missing for all operators
### 9. Signed-zero tracking not standardized
### 10. `max_abs_diff: NaN` in Mul/Add torch results when bitwise_equal=true
