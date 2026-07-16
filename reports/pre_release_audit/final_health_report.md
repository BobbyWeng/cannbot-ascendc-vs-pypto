# Pre-Release Final Health Report

**Project**: cannbot_ascendc_vs_pypto
**Audit Time**: 2026-07-16T23:45:00Z
**Mode**: Comprehensive cross-verification with report corrections

---

## 1. Operator Inventory Summary

| # | Operator | Category | Overall Status | Torch | Ascend C | PyPTO |
|---|----------|----------|---------------|-------|----------|-------|
| 1 | relu | element-wise activation | **COMPLETE** | COMPLETE | TRUE_DEVICE | SUCCESS |
| 2 | mul | element-wise arithmetic | **COMPLETE** | COMPLETE | TRUE_DEVICE | SUCCESS |
| 3 | add | element-wise arithmetic | **COMPLETE** | COMPLETE | TRUE_DEVICE | SUCCESS |
| 4 | div | broadcast arithmetic | **COMPLETE_WITH_LIMITATION** | COMPLETE | TRUE_DEVICE | BLOCKED_BACKEND |
| 5 | equal | element-wise comparison | **COMPLETE_WITH_LIMITATION** | COMPLETE | TRUE_DEVICE | BLOCKED_BACKEND_EQUAL |
| 6 | not | element-wise logical | **REPORT_OUTDATED** | COMPLETE | TRUE_DEVICE | PASS |
| 7 | or | element-wise logical | **REPORT_OUTDATED** | COMPLETE | TRUE_DEVICE | PARTIAL (bitwise_or) |
| 8 | where | element-wise select | **COMPLETE_WITH_LIMITATION** | COMPLETE | TRUE_DEVICE_SCALAR | BLOCKED_BACKEND_WHERE |
| 9 | expand | broadcast/reshape | **INCOMPLETE** | PARTIAL B1 | HOST_PRECOMPUTE | PARTIAL |
| 10 | transpose | permutation | **INCOMPLETE** | PARTIAL B1 | HOST_PRECOMPUTE | BLOCKED (large) |
| 11 | reduce_sum | reduction | **INCOMPLETE** | PARTIAL B1 | HOST_PRECOMPUTE | SUCCESS (no corr) |

## 2. Status Counts

| Status | Count | Operators |
|--------|-------|-----------|
| COMPLETE | 3 | relu, mul, add |
| COMPLETE_WITH_LIMITATION | 3 | div, equal, where |
| REPORT_OUTDATED | 2 | not, or |
| INCOMPLETE | 3 | expand, transpose, reduce_sum |

## 3. Ascend C Implementation Authenticity

### TRUE_DEVICE_IMPLEMENTATION (7 operators)
| Operator | Kernel API | Evidence |
|----------|-----------|----------|
| relu | AscendC::Relu | msprof KERNEL_AIVEC |
| mul | AscendC::Mul | msprof KERNEL_AIVEC |
| add | 3x AscendC::Add | msprof KERNEL_AIVEC |
| div | AscendC::Div + Duplicate | msprof KERNEL_AIVEC |
| equal | Compare + scalar bit expansion | correctness PASS all batches |
| not | Cast + Sub (FP16 NOT emulation) | binary exists, source valid |
| or | Cast + Max (FP16 OR emulation) | binary exists, source valid |

### TRUE_DEVICE_WITH_SCALAR_FALLBACK (1 operator)
| Operator | Issue |
|----------|-------|
| where | per-element scalar loop for Select (37x slower) |

### HOST_PRECOMPUTE_FALLBACK (3 operators)
| Operator | What's on CPU |
|----------|--------------|
| expand | Broadcast expansion — kernel is identity copy |
| transpose | Full permutation — kernel is identity copy |
| reduce_sum | FP32 accumulation pre-reduction — kernel reduces already-reduced data |

## 4. Correctness Summary

| Operator | Torch | Ascend C | PyPTO | Notes |
|----------|-------|----------|-------|-------|
| relu | PASS all B | PASS all B | PASS all B | — |
| mul | PASS all B | PASS all B | PASS all B | — |
| add | PASS all B | PASS all B | PASS all B | — |
| div | PASS all B | PASS all B | BLOCKED_BACKEND | atol=1e-3 rtol=1e-3 |
| equal | PASS all B | PASS all B | BLOCKED_BACKEND | — |
| not | PASS all B | FAIL (script bug) | PASS | Reports previously claimed PASS |
| or | PASS all B | FAIL (script bug) | PARTIAL (bitwise_or) | Reports previously claimed PASS |
| where | PASS all B | PASS all B | BLOCKED_BACKEND | — |
| expand | B=1 only | NOT RUN | NOT RUN | — |
| transpose | B=1 only | NOT RUN | NOT RUN | — |
| reduce_sum | B=1 only | NOT RUN | NOT RUN | — |

## 5. Profiler Summary

| Operator | Torch | Ascend C | PyPTO | Standard |
|----------|-------|----------|-------|----------|
| relu | msprof | msprof | msprof | ✅ |
| mul | msprof | msprof | msprof | ✅ |
| add | msprof | msprof | msprof | ✅ |
| div | msprof (B=32) | msprof (B=32) | N/A | ⚠️ Partial |
| equal | torch.npu.Event | aclrtEvent | N/A | ❌ NOT_COMPARABLE |
| not | torch.npu.Event | aclrtEvent | torch.npu.Event | ❌ NOT_COMPARABLE |
| or | torch.npu.Event | aclrtEvent | torch.npu.Event | ❌ NOT_COMPARABLE |
| where | torch.npu.Event | aclrtEvent | N/A | ❌ NOT_COMPARABLE |
| expand | N/A | N/A | N/A | ❌ Missing |
| transpose | N/A | N/A | N/A | ❌ Missing |
| reduce_sum | N/A | N/A | N/A | ❌ Missing |

## 6. Report Corrections Made

1. **release_summary.json** — PyPTO primary_compute_kernel_us corrected from KERNEL_AICPU (~3ms) to KERNEL_MIX_AIC (50-150us)
2. **release_summary.md** — Performance table corrected; known limitations expanded
3. **README.md (root)** — Expanded from 4 to 11 operators with honest status
4. **operators/README.md** — Corrected from overstated COMPLETE to REPORT_OUTDATED/INCOMPLETE
5. **dashboard/dashboard.json** — Expanded to all 11 operators with correct status
6. **reports/operator_summary.md** — Expanded to all 11 operators with correct status
7. **not/reports/final/final_comparison.json** — Correctness: PASS → FAIL
8. **or/reports/final/final_comparison.json** — Correctness: PASS → FAIL
9. **expand/reports/final/final_comparison.json** — Status: COMPLETE → INCOMPLETE
10. **transpose/reports/final/final_comparison.json** — Status: COMPLETE_WITH_LIMITATION → INCOMPLETE
11. **reduce_sum/reports/final/final_comparison.json** — Status: COMPLETE → INCOMPLETE

## 7. Remaining Issues

### P0
1. **Not/Or Ascend C correctness**: Fix script bug (wrong filename pattern) and re-run correctness.py
2. **Expand/Transpose/ReduceSum**: Need genuine device-side Ascend C kernels or honest HOST_PRECOMPUTE documentation

### P1
3. **Not/Or/Where/Equal msprof**: Standardize profiling to msprof
4. **Div per-batch profiler**: Collect msprof for B=1,2,4,8,16
5. **Expand/Transpose/ReduceSum torch correctness**: Re-run for all 7 batches
6. **Or PyPTO**: Fix bitwise_or → logical_or

### P2
7. **Archive slim for relu/mul/add/div**: Create proper v1 archives
8. **ULP measurement**: Add to correctness checking
9. **Standardize Ascend C warmup/loops/repeat** across all operators

## 8. GitHub Readiness

**YES** — Core is ready. All 11 operators correctly inventoried. Reports corrected. No secrets detected.
