# Profiler & Benchmark Issues (Pre-Release Audit)

## Methodology Violations

### P0 — Mislabeled PyPTO primary compute kernel in release_summary.json
- **Issue**: `release_summary.json` reports PyPTO `primary_compute_kernel_us` as the KERNEL_AICPU one-time init event (~3ms) instead of the KERNEL_MIX_AIC compute kernel
- **Affected**: All 4 arithmetic operators (relu, mul, add, div)
- **Example**: relu B=1 shows 3027.82us for PyPTO primary; actual MIX_AIC is 51.887us
- **Impact**: Makes PyPTO look ~30x slower than it actually is for compute
- **Fix**: Corrected in pre_release_audit; needs release_summary.json update

### P0 — Add final_comparison.json PyPTO primary kernel
- **Fix applied**: The `operators/add/reports/final/final_comparison.json` now correctly shows KERNEL_MIX_AIC primary (136.03us) instead of KERNEL_AICPU (3161.403us)

### P1 — Not/Or/Where/Equal: No msprof profiling
- All four use torch.npu.Event or aclrtEvent instead of msprof
- Latency numbers are host-synchronized operation times, NOT `primary_compute_kernel_us`
- Marked as NOT_COMPARABLE — cannot rank with relu/mul/add/div

### P1 — Expand/Transpose/ReduceSum: No profiling at all
- Expand/Transpose/ReduceSum have no profiler data
- Ascend C implementations are host precompute fallback, not device kernels

### P2 — Warmup mismatch for all Ascend C operators
- All Ascend C kernels use warmup=100, loops=1000 (vs standard 200/100)
- Div additionally uses repeat=10 (vs standard 5)
- Difference is minor but should be standardized

## Report vs Profiler Discrepancies

### Add final_comparison.json — CORRECTED
- Before: PyPTO primary = 3161.403us (KERNEL_AICPU init)
- After: PyPTO primary = 136.03us (KERNEL_MIX_AIC compute)
- Source: `operators/add/reports/parsed/pypto_b1.json`

### Release summary PyPTO numbers — CORRECTED
- All PyPTO `primary_compute_kernel_us` values corrected from KERNEL_AICPU to KERNEL_MIX_AIC
- See corrected dashboard.json and final_health_report for accurate values

## Missing Coverage

### Div per-batch profiler
- Only B=32 has msprof data
- B=1,2,4,8,16 missing from msprof (only have aclrtEvent timing)

### Expand/Transpose/ReduceSum
- Zero profiler data exists
- Cannot confirm device-side kernel performance
