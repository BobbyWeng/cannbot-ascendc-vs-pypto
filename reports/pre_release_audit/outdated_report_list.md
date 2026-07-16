# Outdated Reports — Pre-Release Audit Corrections

## CORRECTED Reports

### 1. release_summary.json
- **Issue**: PyPTO primary_compute_kernel_us mislabeled as KERNEL_AICPU init (~3ms) instead of KERNEL_MIX_AIC compute (~50-150us)
- **Fix**: Regenerated with correct KERNEL_MIX_AIC values
- **File**: `reports/project_release/release_summary.json`

### 2. release_summary.md
- **Issue**: Performance table for B=32 shows wrong PyPTO values (e.g. relu 98.88us compute vs actual 103.909us)
- **Fix**: Updated table with correct values
- **File**: `reports/project_release/release_summary.md`

### 3. Root README.md
- **Issue**: Only covers 4 operators; shows add/mul/div as "planned"; div as "Needs HW"
- **Fix**: Expand to all 11 operators with correct status
- **File**: `README.md`

### 4. operators/README.md
- **Issue**: Shows not/or/expand/reduce_sum as COMPLETE; transpose as COMPLETE_WITH_LIMITATION
- **Fix**: Corrected to actual status (not/or=REPORT_OUTDATED, expand/reduce_sum=INCOMPLETE, transpose=INCOMPLETE)
- **File**: `operators/README.md`

### 5. Dashboard (dashboard.json + index.html)
- **Issue**: Only has equal/where/not/or — missing relu/mul/add/div; not/or marked COMPLETE
- **Fix**: Regenerated with all 11 operators, correct statuses
- **Files**: `dashboard/dashboard.json`, `dashboard/index.html`

### 6. operators/not/reports/final/final_comparison.json
- **Issue**: Claims Ascend C correctness PASS but stored JSON shows FAIL
- **Fix**: Updated correctness field to show actual FAIL status

### 7. operators/or/reports/final/final_comparison.json
- **Issue**: Same as not — claims PASS but JSON shows FAIL
- **Fix**: Updated correctness field

### 8. reports/operator_summary.md
- **Issue**: Only covers equal/where/not/or; claims not/or/expand as COMPLETE
- **Fix**: Expanded to all 11 operators

### 9. reports/operator_summary.json
- **Same issue as operator_summary.md**
- **Fix**: Expanded to all 11 operators

## Reports that remain CORRECT

- operators/relu/reports/final/ — All data consistent
- operators/mul/reports/final/ — All data consistent
- operators/add/reports/final/ — CORRECTED (PyPTO primary kernel)
- operators/div/reports/final/ — All data consistent
- operators/where/reports/final/ — Accurate (COMPLETE_WITH_LIMITATION)
- operators/equal/reports/final/ — Accurate (COMPLETE_WITH_LIMITATION)

## Reports that remain PROVISIONAL (correctly)

- operators/expand/reports/final/ — Marked as INCOMPLETE
- operators/transpose/reports/final/ — Marked as INCOMPLETE  
- operators/reduce_sum/reports/final/ — Marked as INCOMPLETE
