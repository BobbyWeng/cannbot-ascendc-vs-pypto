# Remaining Work — Validation Freeze

Issues found during audit that should be addressed before any future release.

## P0: Must Fix

### 1. msprof for Logical Operators (equal, not, or, where)

These 4 operators lack msprof device-kernel profiling. Without it, they are NOT_COMPARABLE with arithmetic operators. This blocks their inclusion in the three-way performance ranking.

**Task**: Run `bash benchmark/run_all.sh` for each operator with msprof enabled.

### 2. Div Torch Correctness — B=4,8,16,32 Reference Files

Div torch correctness skips 4 of 6 declared batches due to missing reference files. The `all_pass: true` flag is misleading.

**Task**: Generate reference BINs for B=4,8,16,32 and re-run torch correctness.

### 3. SKILL_TRACE.md Creation

No operator has a SKILL_TRACE.md. Required by project policy for auditability.

**Task**: Create SKILL_TRACE.md for each operator's ascendc/ directory.

## P1: Should Fix

### 4. SHA256SUMS for Missing Operators

Only 4/12 operators have valid SHA256SUMS. 5 have empty/broken files, 3 are missing entirely.

**Task**: Generate SHA256SUMS for all 12 operators following the archive policy (source + config + reports, no build artifacts, relative paths).

### 5. Mul final_comparison.json Stale Value

The final comparison JSON for mul shows torch B1=3.876us but parsed data shows 9.0us. While the dashboard uses the correct value, the on-disk report is stale.

**Task**: Regenerate mul's final comparison reports from authoritative parsed data.

### 6. operator_summary.md/json Sync with current_release.json

The operator_summary files showed 11 operators (missing matmul) and 3 COMPLETE (should be 4). While identified and noted in this audit, the on-disk files need regeneration.

**Task**: Regenerate operator_summary.md and operator_summary.json from current_release.json.

## P2: Should Do

### 7. Reduce_sum Ascend C FP16 Accum Precision

Reduce_sum achieves only 21/70 PASS for Ascend C and PyPTO due to FP16 accumulation. Consider implementing FP32 accumulation or documenting the expected precision.

### 8. Equal Ascend C Kernel Optimization

The equal kernel uses per-element scalar GetValue/SetValue for bit expansion. This is a known anti-pattern. Consider using vectorized Select or mask operations.

### 9. Where Ascend C Kernel Vectorization

Where kernel uses per-element scalar selection. Could be replaced with vectorized Select or ternary operations.

### 10. PyPTO Orchestrator State File Restoration

5 operators (add, equal, not, or, where) have artifact_manifest.json files falsely claiming `.orchestrator_state.json` exists. Need to either restore the state files or correct the manifests.

## P3: Enhancement

### 11. Dashboard Regeneration

Dashboard needs regeneration using `python dashboard.py --release reports/release/current_release.json` to ensure operator count (12) and status counts (4 COMPLETE) are correct.

### 12. README Sync

Update project README to reflect v1.1 status changes (not, expand, transpose, reduce_sum status changes; matmul addition).

### 13. Archive Release

Generate formal archives for the 4 operators without them (expand, transpose, reduce_sum, matmul).

## Summary

| Priority | Items | Effort | Impact |
|----------|-------|--------|--------|
| P0 | 3 | Medium | Blocks comparability + auditability |
| P1 | 3 | Low | Data consistency |
| P2 | 4 | Medium | Performance + compliance |
| P3 | 3 | Low | Documentation |
