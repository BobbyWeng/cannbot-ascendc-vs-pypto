# PyPTO Full Audit Report
**Generated**: 2026-07-23T11:12:12Z
**Branch**: codex/pypto-full-audit
**Environment**: torch 2.8.0+cpu, torch_npu 2.8.0.post2, CANN 9.0.0, PyPTO d1c290f36

## 14-Operator Classification (Audit 2026-07-23)

### PYPTO_NATIVE_PASS (8 operators)
All 8 verified B1-B64 bitwise correctness + KERNEL_MIX_AIC profiling under current env.

| # | Operator | Semantic   | Correctness | Kernels/Call | Primary Kernel | Full Call   |
|---|----------|------------|-------------|--------------|----------------|-------------|
| 1 | add      | ELEMENTWISE| B1-B64 BITWISE | 3         | 38.6us         | ~1231us     |
| 2 | equal    | LOGICAL    | B1-B64 BITWISE | 1         | —              | —           |
| 3 | not      | LOGICAL    | B1-B64 BITWISE | 1         | —              | —           |
| 4 | or       | LOGICAL    | B1-B64 BITWISE | 1         | —              | —           |
| 5 | where    | LOGICAL    | B1-B64 BITWISE | 1         | —              | —           |
| 6 | div      | ELEMENTWISE| B1-B64 BITWISE | 1         | —              | —           |
| 7 | transpose| LAYOUT     | B1-B64 BITWISE | B (batch-varies) | 47.8us    | ~10204us(B64)|
| 8 | reduce_sum| REDUCTION  | B1-B64 atol=0.02 | 2         | 42.6us(B1)     | —           |

### TORCH_FALLBACK (1 operator)
Excluded from PyPTO PASS count, G8 coverage, and performance ranking.

| # | Operator | Reason |
|---|----------|--------|
| 9 | expand   | expand_wrapper uses x.expand().clone() — pure torch_npu, zero KERNEL_MIX_AIC observed. expand_row JIT kernel exists but is not the active code path. |

### UNDER_INVESTIGATION_JIT_REGRESSION (2 operators)
Historical Stage 5-7 completed. Current env: B=1 PASS, B>1 CompileFunction error.

| #  | Operator | Historical   | Current B=1 | Current B>1 |
|----|----------|-------------|-------------|-------------|
| 10 | relu     | Stage 7 PASS | PASS        | FAIL (CompileFunction) |
| 11 | mul      | Stage 7 PASS | PASS        | FAIL (CompileFunction) |

### UNDER_INVESTIGATION (3 operators)
Cannot compile in current env. AGENTS.md criteria for BLOCKED_BACKEND not yet met.

| #  | Operator  | Error | Criteria Met |
|----|-----------|-------|-------------|
| 12 | softmax   | CompileFunction INTERNAL BUG | 0/6 |
| 13 | layernorm | CompileFunction INTERNAL BUG | 0/6 |
| 14 | matmul    | Compile succeeds, max_diff=95.77 | 0/6 |

AGENTS.md criteria for BLOCKED_BACKEND (6 items): version matrix, official examples, minimal success/fail, frontend IR, failing pass, 3+ candidates.

## Critical Audit Findings

### expand: TORCH_FALLBACK (NOT PyPTO JIT)
- msprof: only KERNEL_AIVEC BroadcastTo kernel, ZERO KERNEL_MIX_AIC
- Implementation: x.expand().clone() — torch_npu native path
- expand_row JIT kernel exists as reference only
- **Excluded from all PyPTO rankings**

### transpose: Per-Batch Kernel Launch
- B=64: 64 separate JIT kernel launches per logical call
- Primary kernel ~47.8us per batch row
- Full call ~10204us (64 MIX_AIC + AICPU overhead)

### add: 3 Binary Kernels Per Call
- 4 inputs = 3 chained binary adds
- Primary ~38.6us each, full call ~1231us

### relu/mul: JIT State Regression
- B=1 standalone: PASS
- B=2 or B=64 standalone: FAIL (CompileFunction: Run pass failed)
- Hypothesis: tile_fwk global state corruption from accumulated profiling sessions
- Investigation: Phase 3 isolation matrix pending

### softmax/layernorm/matmul: Awaiting JIT Fix
- Blocked by shared tile_fwk JIT regression
- Will re-test after relu/mul diagnosis
- BLOCKED_BACKEND classification premature (criteria not met)

## State File Corrections (2026-07-23)
- [x] expand: removed from PyPTO PASS, reclassified TORCH_FALLBACK
- [x] relu/mul: reclassified UNDER_INVESTIGATION_JIT_REGRESSION, history preserved
- [x] softmax/layernorm/matmul: BLOCKED_BACKEND → UNDER_INVESTIGATION, historical_blockers archived
- [x] All 14 orchestrator_state.json updated with audit_2026_07_23_classification

## Next Actions
1. **Phase 3**: ReLU/Mul 10-matrix JIT isolation test
2. **Phase 4**: Fix or workaround (3 candidates max)
3. **Phase 5**: Softmax/LayerNorm/MatMul re-investigation
4. Release/dashboard deferred until all investigations complete

---
*Full evidence: /tmp/opencode/audit_*/*. Full state: operators/*/pypto/.orchestrator_state.json*
