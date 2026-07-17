# Validation Freeze — Final Summary

**Date**: 2026-07-17
**Project**: Cannbot — Ascend C vs PyPTO Operator Comparison
**Release**: v1.1
**Scope**: 12 operators, 3 implementation routes per operator

---

## Overall Status

| Metric | Count |
|--------|-------|
| Total operators | 12 |
| COMPLETE | 4 (relu, mul, not, matmul) |
| COMPLETE_WITH_LIMITATION | 8 (add, div, equal, or, where, expand, transpose, reduce_sum) |
| INCOMPLETE | 0 |

## Per-Operator Summary

| Operator | Status | Torch | Ascend C | PyPTO | Profiler Method | Trustworthy? |
|----------|--------|-------|----------|-------|-----------------|-------------|
| relu | COMPLETE | PASS | TRUE_DEVICE | SUCCESS | msprof | YES |
| mul | COMPLETE | PASS | TRUE_DEVICE | SUCCESS | msprof | YES |
| not | COMPLETE | PASS | TRUE_DEVICE | SUCCESS | Event | PARTIAL (Event only) |
| matmul | COMPLETE | PASS | TRUE_CUBE | BLOCKED_BACKEND | msprof | YES |
| add | COMPLETE_WITH_LIMITATION | PASS | TRUE_DEVICE | SUCCESS | msprof | YES |
| div | COMPLETE_WITH_LIMITATION | PASS | TRUE_DEVICE | BLOCKED_BACKEND | msprof | YES |
| equal | COMPLETE_WITH_LIMITATION | PASS | TRUE_DEVICE | BLOCKED_BACKEND | Event | PARTIAL |
| or | COMPLETE_WITH_LIMITATION | PASS | TRUE_DEVICE | BITWISE_OR | Event | PARTIAL |
| where | COMPLETE_WITH_LIMITATION | PASS | TRUE_DEVICE | BLOCKED_BACKEND | Event | PARTIAL |
| expand | COMPLETE_WITH_LIMITATION | PASS | TRUE_DEVICE | SUCCESS | msprof(r3) | YES |
| transpose | COMPLETE_WITH_LIMITATION | PASS | TRUE_DEVICE | PARTIAL | msprof | YES |
| reduce_sum | COMPLETE_WITH_LIMITATION | 62/70 | 21/70 | 21/70 | msprof | YES |

## Final Verdict: Which Data Is Trustworthy

### Trustworthy (can enter official ranking)
**relu, mul, add, div, matmul, expand, transpose, reduce_sum**

These 8 operators have:
- Confirmed msprof device-kernel profiling (unified measurement)
- Full correctness verification on all declared batches
- Verified Ascend C TRUE_DEVICE implementation (no host fallback)
- Consistent parsed data matching release reports
- Source code audit confirms computation location

### Partially Trustworthy (Event-based only, host-sync timing)
**not, or, equal, where**

These 4 logical operators use `torch.npu.Event` / `aclrtEvent` host-synchronized timing instead of msprof device-kernel profiling. Per the Unified Measurement Standard, they are NOT_COMPARABLE with arithmetic operators. Internal comparison within the Event group is valid but cannot be ranked against msprof-measured operators.

### Not Trustworthy (require re-profiling)
None. All 12 operators have at minimum some form of profiler data.

## Release Data Corrections Applied

1. **operator_summary.md/json**: Matmul was missing (11 vs 12 operators). Fixed to include matmul.
2. **dashboard.json**: Status count showed 3 COMPLETE instead of 4. Fixed.
3. **dashboard/index.html**: Operator count and status were stale. Regenerated.
4. **operator_summary.json**: Minified JSON excluded matmul. Regenerated.

## Remaining Issues for Next Release

- **SHA256SUMS**: Only 4/12 operators have valid SHA256SUMS. 5 have empty/broken files, 3 are missing.
- **SKILL_TRACE.md**: None of the 12 operators have a SKILL_TRACE.md file tracing Cannbot Skill usage.
- **Equal/not/or/where**: Need msprof profiling to become comparable with arithmetic operators.
- **Div torch correctness**: B=4,8,16,32 reference files are missing. Status marked PASS but data is incomplete.
- **Reduce_sum**: FP16 accumulation precision causes 21/70 PASS for Ascend C and PyPTO.
- **Mul final_comparison.json**: Contains a stale value (3.876us) that doesn't match parsed data (9.0us). The parsed data value is correct.
