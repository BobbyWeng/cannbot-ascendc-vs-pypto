# Skill Trace Summary — RC2 Audit

**Generated**: 2026-07-17T00:00:00Z  
**Scope**: All 12 operators in `operators/`

## Matrix

| Operator | Ascend C | PyPTO | Orchestrator State | Notes |
|----------|----------|-------|-------------------|-------|
| add | LEGACY_UNVERIFIED_SKILL_USAGE | NON_COMPLIANT | missing | BLOCKED_BACKEND |
| div | LEGACY_UNVERIFIED_SKILL_USAGE | NON_COMPLIANT | exists_at_root | BLOCKED_BACKEND; state at operator root, not in pypto/ |
| equal | LEGACY_UNVERIFIED_SKILL_USAGE | NON_COMPLIANT | missing | BLOCKED_BACKEND |
| expand | LEGACY_UNVERIFIED_SKILL_USAGE | NON_COMPLIANT | missing | BLOCKED_BACKEND; SPEC/API_REPORT/DESIGN dirs empty |
| matmul | LEGACY_UNVERIFIED_SKILL_USAGE | NON_COMPLIANT | missing | BLOCKED_BACKEND; flat file layout (no subdirs) |
| mul | LEGACY_UNVERIFIED_SKILL_USAGE | LEGACY_UNVERIFIED_SKILL_USAGE | exists_in_pypto | State present |
| not | LEGACY_UNVERIFIED_SKILL_USAGE | NON_COMPLIANT | missing | |
| or | LEGACY_UNVERIFIED_SKILL_USAGE | NON_COMPLIANT | missing | |
| reduce_sum | LEGACY_UNVERIFIED_SKILL_USAGE | LEGACY_UNVERIFIED_SKILL_USAGE | exists_in_pypto | State present |
| relu | LEGACY_UNVERIFIED_SKILL_USAGE | LEGACY_UNVERIFIED_SKILL_USAGE | exists_in_pypto | State present |
| transpose | LEGACY_UNVERIFIED_SKILL_USAGE | NON_COMPLIANT | exists_in_pypto | BLOCKED_BACKEND; SPEC/API_REPORT/DESIGN dirs empty |
| where | LEGACY_UNVERIFIED_SKILL_USAGE | NON_COMPLIANT | missing | BLOCKED_BACKEND |

## Classification Breakdown

### Ascend C (12/12)
- **LEGACY_UNVERIFIED_SKILL_USAGE**: 12 operators
- All confirmed by code pattern (kernel.asc, host.asc, tiling.h, CMakeLists.txt)
- Skills identified: `ascendc-kernel-develop-workflow`, `ascendc-api-best-practices`, `ascendc-tiling-design`

### PyPTO (12/12)
- **LEGACY_UNVERIFIED_SKILL_USAGE**: 4 operators (mul, reduce_sum, relu, transpose)
- **NON_COMPLIANT**: 8 operators (add, div, equal, expand, matmul, not, or, where)
  - 6 missing `.orchestrator_state.json` entirely
  - 1 has state at root level not in pypto/ (div)
  - 2 have empty SPEC/API_REPORT/DESIGN dirs (expand, transpose)

## Key Findings

1. **No SKILL_TRACE files existed before this audit** — 0 out of 12 operators had trace documentation.
2. **All Ascend C implementations** show consistent LEGACY_UNVERIFIED_SKILL_USAGE with clear code pattern evidence.
3. **PyPTO compliance gap**: 8/12 operators are NON_COMPLIANT due to missing orchestrator state or empty artifact directories.
4. **Backend blockers**: 7 operators (add, div, equal, expand, matmul, transpose, where) have BLOCKED_BACKEND issues.

## Recommendations

1. Add `.orchestrator_state.json` to all NON_COMPLIANT PyPTO operator directories.
2. Populate empty SPEC/API_REPORT/DESIGN directories for expand and transpose.
3. Standardize orchestrator state location to operator root per AGENTS.md convention.
4. Migrate all LEGACY_UNVERIFIED_SKILL_USAGE entries to COMPLIANT by recording formal traces for future development.
