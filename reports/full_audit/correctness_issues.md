# Correctness Issues Found

## P0 - Must Fix

### 1. Not Ascend C correctness never properly verified
- **Operator**: not
- **File**: `operators/not/reports/correctness/ascendc_correctness.json`
- **Issue**: All 7 batches show FAIL with `[Errno 2] No such file or directory: .../reference_b1_bool.bin'`
- **Root cause**: Correctness script looks for `reference_b*_bool.bin` but actual files are named `reference_b*_{case}.bin` (e.g., `reference_b1_random_mask.bin`)
- **Reports claim**: Ascend C correctness PASS (contradicts stored JSON)
- **Impact**: Ascend C Not correctness is unverified. Script bug, not kernel bug — files exist but wrong pattern.

### 2. Or Ascend C correctness never properly verified
- **Operator**: or
- **File**: `operators/or/reports/correctness/ascendc_correctness.json`
- **Issue**: Same script bug as Not — looks for `reference_b*_bool.bin` but files are named `reference_b*_{case}.bin`
- **Reports claim**: Ascend C correctness PASS (contradicts stored JSON)

### 3. Or PyPTO uses bitwise_or instead of logical_or
- **Operator**: or
- **Route**: pypto/src/or_impl.py
- **Issue**: Uses `pypto.bitwise_or` instead of `pypto.op.logical_or`
- **Impact**: For non-0/1 uint8 inputs, results differ from spec. Only passes because test data is strictly 0/1.
- **Note**: The report itself acknowledges this in a caveat but still claims PASS

### 4. Add final_comparison.json has wrong PyPTO primary kernel
- **Operator**: add
- **File**: `operators/add/reports/final/final_comparison.json`
- **Issue**: PyPTO B=1 `primary_compute_kernel_us: 3161.403` is KERNEL_AICPU (init event ~3ms), not KERNEL_MIX_AIC (136.03 us)
- **Impact**: The primary compute metric is mislabeled — it reports AICPU init time as the primary compute kernel

## P1 - Should Fix

### 5. Div Torch correctness coverage gap
- **Operator**: div
- **File**: `operators/div/torch/correctness_results.json`
- **Issue**: B=1,2 PASS; B=4,8,16,32 SKIP (reference files didn't exist at time of run)
- **Fix**: Re-run torch correctness — reference files now exist

### 6. Div special value failures undocumented
- **Operator**: div
- **Route**: ascendc
- **Issue**: Special value cases (div-by-zero, inf/NaN) expectedly fail due to FP16 hardware differences but this is not documented in final report
- **Fix**: Add note documenting why special values fail

### 7. PyPTO equal correctness blocked but orchestrator state missing
- **Operator**: equal
- **Route**: pypto
- **Issue**: No `.orchestrator_state.json` — can't verify what orchestrator stages completed
- **Fix**: Create orchestrator state file documenting BLOCKED_BACKEND state

### 8. Expand torch correctness only B=1
- **Operator**: expand
- **File**: `operators/expand/torch/correctness_results.json`
- **Issue**: Only B=1 data present but claims `all_pass: true`
- **Fix**: Re-run for all 7 batches

### 9. Transpose torch correctness only B=1
- **Operator**: transpose
- **Same issue as expand**

### 10. Reduce_sum torch correctness only B=1
- **Operator**: reduce_sum
- **Same issue; additionally shows fp32_accum fails on NaN case for B=1**

## P2 - Can Do Later

### 11. ULP measurement missing for all operators
- **Issue**: Only bitwise comparison; no ULP-based tolerance
- **Fix**: Add to common/correctness/

### 12. Signed-zero tracking not standardized
- **relu**: Exempted (documented)
- **mul**: Tracked (strict)
- **add**: Tracked (strict)
- **div**: Not applicable (tolerance-based)
- **Fix**: Standardize in spec
