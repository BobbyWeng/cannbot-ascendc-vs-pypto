# Historical Reports

This directory contains reports and artifacts that have been moved out of `reports/` for cleanup purposes.

## Policy

- Contents in this directory are **for historical tracking only**.
- They do **not** represent the current state of the project.
- They do **not** participate in the Dashboard or release summary.
- Current authoritative state is in `reports/release/current_release.json`.

## What was removed

The following directories and files were removed from `reports/` because they contained outdated or superseded audit results:

### Removed (not moved — permanently deleted)

These were internal working documents whose conclusions have been superseded by current source truth:

- `reports/full_audit/` — 20 files: operator inventory, cleanup plan, outdated matrix
- `reports/pre_release_audit/` — 16 files: GitHub readiness, outdated matrices, pycache
- `reports/project_audit/` — 3 files: audit of only 4 operators
- `reports/project_release/` — 6 files: separate release summary (integrated into current_release.json)
- `reports/project_repair/` — 5 files: repair summary for past state (no longer relevant)
- `reports/device_side_final_report.md` — Plan document superseded by current source
- `reports/device_side_plan.md` — Plan document superseded by current source
- `reports/project_audit_after.json` / `.md` — Old project snapshot (58 files/464KB)
- `reports/cleanup_manifest.json` — Old manifest from initial project migration
- `reports/full_audit.tar.gz` — Compressed archive of deleted full_audit
- `reports/runtime/` — Empty directory

### Batch-level removed from `reports/batches/layout_reduce_v1/`

- pypto_jit_matrix.csv, pypto_jit_path_diff.csv
- pypto_success_pattern_audit.json/.md
- transpose_optimization_report.json/.md
- FINAL_ANSWERS.md
- key_blocker_repair_summary.json/.md
- profiler_completion_report.json/.md
- full_profiler_summary.md
- profiler_summary.md

### Operator-level removed

- `operators/expand/reports/final/final_comparison.json` — Falsely claimed HOST_PRECOMPUTE_FALLBACK (source is TRUE_DEVICE)
- `operators/transpose/reports/final/final_comparison.json` — Same
- `operators/reduce_sum/reports/final/final_comparison.json` — Same
- Corresponding `.md` files for above

## Key correction

The most significant correction from this cleanup:

**Expand/Transpose/ReduceSum Ascend C kernels were HOST_PRECOMPUTE_FALLBACK → TRUE_DEVICE_IMPLEMENTATION**

Source code audit confirmed that all three have genuine device-side kernels:
- Expand: `GetValue + Duplicate[384]`
- Transpose: `DataCopyPad + element-wise swap + DataCopyPad out`
- ReduceSum: `ReduceSum<half> Level 2`

Previous reports that claimed "identity copy with host precompute" were incorrect.
