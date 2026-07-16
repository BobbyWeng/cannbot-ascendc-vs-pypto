# Outdated Reports

## Release Summary Reports

### release_summary.json
**Issue**: PyPTO primary_compute_kernel_us everywhere is mislabeled as KERNEL_AICPU
**Example**: relu B=1 shows `primary_compute_kernel_us: 3027.82` (the AICPU init event) instead of `51.128` (the MIX_AIC compute kernel)
**Fix**: Correct to use KERNEL_MIX_AIC mean for primary compute, keep AICPU in aicpu_executor field
**File**: `reports/project_release/release_summary.json`

### release_summary.md
**Issue**: Performance table for B=32 shows PyPTO values that are wrong
- relu PyPTO "compute" shows 98.88 us but release_summary.json says 3698.694
- These numbers don't match each other
**Fix**: Align with parsed profiler data
**File**: `reports/project_release/release_summary.md`

## Operator-Specific Reports

### Not/Or/Where/Equal final reports
**Issue**: All claim msprof methodology but actually used torch.npu.Event / aclrtEvent
**Issue**: No parsed profiler data exists
**Issue**: For Not and Or, claim Ascend C correctness PASS but stored correctness JSON shows FAIL
**Affected files**:
- `operators/not/reports/final/final_comparison.md`
- `operators/not/reports/final/final_comparison.json`
- `operators/not/reports/final/final_comparison.csv`
- `operators/or/reports/final/final_comparison.md`
- `operators/or/reports/final/final_comparison.json`
- `operators/or/reports/final/final_comparison.csv`
- `operators/where/reports/final/final_comparison.md`
- `operators/where/reports/final/final_comparison.json`
- `operators/where/reports/final/final_comparison.csv`
- `operators/equal/reports/final/final_comparison.md`
- `operators/equal/reports/final/final_comparison.json`
- `operators/equal/reports/final/final_comparison.csv`

### Add final_comparison.json
**Issue**: PyPTO primary_compute_kernel_us is the AICPU init event, not the MIX_AIC compute kernel
**File**: `operators/add/reports/final/final_comparison.json`

### Root README.md
**Issue**: Mentions reduce_sum, layer_norm, matmul as "planned" but does NOT list equal/not/or/where/expand/transpose/reduce_sum
**Status table only shows 4 operators (relu, add, mul, div)**
**Fix**: Expand to cover all 11 operators

### Operator Summary (reports/operator_summary.json and .md)
**Issue**: Only covers equal/where/not/or — missing relu/mul/add/div/expand/transpose/reduce_sum
**Issue**: Dashboard already covers more operators but summary is incomplete

## Dashboard

### dashboard.json
**Issue**: Only has equal/where/not/or — missing relu/mul/add/div/expand/transpose/reduce_sum
**File**: `dashboard/dashboard.json`

### dashboard/index.html
**Issue**: Only has logical_ops_v1 batch — missing arithmetic_ops_v1
**Issue**: References v4 archives but root has v2 archives for equal/not/or/where, v2 for div
**Issue**: Archive listings mismatch between dashboards and actual archives
