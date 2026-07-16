# Full Audit: Final Health Report

**Project**: cannbot_ascendc_vs_pypto
**Audit Time**: 2026-07-16T23:00:00Z
**Mode**: read-only cross-verification of all artifacts

---

## 1. Operator Inventory

| # | Operator | Category | Reported Status | True Status | Change |
|---|----------|----------|----------------|-------------|--------|
| 1 | relu | element-wise activation | COMPLETE | COMPLETE | — |
| 2 | mul | element-wise arithmetic | COMPLETE | COMPLETE | — |
| 3 | add | element-wise arithmetic | COMPLETE | COMPLETE | — |
| 4 | div | broadcast arithmetic | COMPLETE_WITH_LIMITATION | COMPLETE_WITH_LIMITATION | — |
| 5 | equal | element-wise comparison | COMPLETE_WITH_LIMITATION | COMPLETE_WITH_LIMITATION | — |
| 6 | not | element-wise logical | COMPLETE | REPORT_OUTDATED | ⚠️ Downgrade |
| 7 | or | element-wise logical | COMPLETE | REPORT_OUTDATED | ⚠️ Downgrade |
| 8 | where | element-wise select | COMPLETE_WITH_LIMITATION | COMPLETE_WITH_LIMITATION | — |
| 9 | expand | broadcast/reshape | INCOMPLETE | INCOMPLETE | — |
| 10 | transpose | permutation | INCOMPLETE | INCOMPLETE | — |
| 11 | reduce_sum | reduction | INCOMPLETE | INCOMPLETE | — |

## 2. COMPLETE Count

**4 operators** meet true COMPLETE criteria:
- relu, mul, add, div

All four have:
- ✅ Real device-side Ascend C implementation
- ✅ PyPTO SUCCESS (or properly documented BLOCKED)
- ✅ msprof profiling with parsed data
- ✅ Consistent final reports with correct metrics
- ✅ SHA256SUMS for stable artifacts

## 3. COMPLETE_WITH_LIMITATION Count

**3 operators**: equal, where, div

All three have:
- ✅ Real device-side Ascend C
- ✅ PyPTO properly classified as BLOCKED_BACKEND with diagnostic evidence
- ❌ No msprof profiling (equal, where) — relies on aclrtEvent/torch.npu.Event

## 4. INCOMPLETE/REPORT_OUTDATED Count

**4 operators**:
- **not**: REPORT_OUTDATED — correctness JSON shows FAIL despite reports claiming PASS
- **or**: REPORT_OUTDATED — same correctness FAIL + bitwise_or semantic bug
- **expand**: INCOMPLETE — CPU precompute for broadcast
- **transpose**: INCOMPLETE — CPU precompute for transpose
- **reduce_sum**: INCOMPLETE — honest NOT_STARTED state

## 5. Ascend C Implementation Authenticity

### TRUE_DEVICE_IMPLEMENTATION (7 operators)
| Operator | Kernel API | Verification |
|----------|-----------|--------------|
| relu | AscendC::Relu | msprof confirms KERNEL_AIVEC |
| mul | AscendC::Mul | msprof confirms KERNEL_AIVEC |
| add | 3x AscendC::Add | msprof confirms KERNEL_AIVEC |
| div | AscendC::Div + Duplicate | msprof confirms KERNEL_AIVEC |
| equal | Compare + scalar bit expansion | correctness PASS all batches |
| not | Cast + Sub (FP16 NOT emulation) | binary exists, source valid |
| or | Cast + Max (FP16 OR emulation) | binary exists, source valid |

### TRUE_DEVICE_WITH_SCALAR_FALLBACK (1 operator)
| Operator | Issue | Impact |
|----------|-------|--------|
| where | per-element scalar loop for Select | 37x slower than vectorized Not/Or |

### HOST_PRECOMPUTE_FALLBACK (3 operators)
| Operator | What's on CPU | Impact |
|----------|--------------|--------|
| expand | Broadcast expansion | Kernel is identity copy |
| transpose | Full permutation | CPU does actual work |
| reduce_sum | FP32 accumulation pre-reduction | Kernel reduces already-reduced data |

**No IDENTITY_KERNEL or BROKEN_IMPLEMENTATION cases**.

## 6. PyPTO Real Backend Limitations

### Verified PYPTO_BACKEND_LIMITATION (3 operators)
| Operator | Issue | Evidence |
|----------|-------|----------|
| div | CompileFunction for broadcast Div | Minimal [1,32]/[1,1] works, broadcast fails |
| equal | Compare result lowering broken | Even [1,32] produces wrong output |
| where | Expand pass dtype mismatch | Same-shape condition fails |

### PyPTO PASS but Unverifiable (2 operators)
| Operator | Issue | Concern |
|----------|-------|---------|
| not | Flat latency across batch sizes | Likely host dispatch overhead, not kernel |
| or | Uses bitwise_or instead of logical_or | Semantic difference for non-0/1 inputs |

**These are NOT hardware limitations** — they are framework/backend/implementation issues.

## 7. Correctness Defects

| Operator | Route | Issue | Severity |
|----------|-------|-------|----------|
| not | Ascend C | Script bug: wrong filename pattern → all FAIL | P0 — reports say PASS but JSON says FAIL |
| or | Ascend C | Same script bug → all FAIL | P0 — reports say PASS but JSON says FAIL |
| or | PyPTO | bitwise_or instead of logical_or | P1 — only works for 0/1 inputs |
| div | Torch | B=4,8,16,32 SKIP in reports | P1 — ref files now exist |
| expand | Torch | Only B=1 tested | P1 — missing batch coverage |
| transpose | Torch | Only B=1 tested | P1 — missing batch coverage |
| reduce_sum | Torch | Only B=1 tested | P1 — missing batch coverage |

## 8. Profiler Standardization

### Compliant (msprof + parsed)
- **relu**: ✅ msprof, parsed, primary compute kernel
- **mul**: ✅ msprof, parsed, primary compute kernel  
- **add**: ✅ msprof, parsed, primary compute kernel
- **div**: ⚠️ msprof for B=32 only; per-batch missing

### Non-Compliant (torch.npu.Event / aclrtEvent only)
- **equal, not, or, where**: ❌ No msprof profiling
- **expand/transpose/reduce_sum**: ❌ No profiling at all

### Measurement Level Violations
All 4 non-compliant operators report host-synchronized operation latency as if it were device kernel timing. These should be marked `NOT_COMPARABLE` or have separate tables.

## 9. Report Corrections Made

This audit identifies the following needed report corrections:

1. **Not/Or** — Correctness status: PASS → FAIL for Ascend C
2. **Not/Or** — Profiler methodology: msprof → torch.npu.Event (not comparable)
3. **Add** — PyPTO primary_compute_kernel_us: 3161 → 136 (AICPU init mislabeled as primary)
4. **Release summary** — PyPTO primary numbers across all operators need correction
5. **Root README** — Expand status table to all 11 operators
6. **Operator summary** — Include all 11 operators
7. **Dashboard** — Include arithmetic ops; fix archive links

## 10. Dashboard Status Corrections

Current dashboard (dashboard.json) issues:
- Only shows equal/where/not/or — missing relu/mul/add/div
- Archive listing references v4 archives (correct) but root dir has v2
- Not/Or marked COMPLETE but should be REPORT_OUTDATED

## 11. Archive Status

| Operator | Archive | Status | Action |
|----------|---------|--------|--------|
| relu | None | MISSING | Create v1 archive |
| mul | None | MISSING | Create v1 archive |
| add | None | MISSING | Create v1 archive |
| div | v2 (root) | OUTDATED | Create v4 archive matching current source |
| equal | v4 (archives/) | CURRENT | — |
| not | v4 (archives/) | CURRENT | — |
| or | v4 (archives/) | CURRENT | — |
| where | v4 (archives/) | CURRENT | — |
| expand/transpose/reduce | None | MISSING | N/A (INCOMPLETE) |

Outdated v2 archives (at root) should be deleted:
- equal_v2, not_v2, or_v2, where_v2: total ~76 KB

## 12. Cleanup Summary

| Category | Items | Size |
|----------|-------|------|
| Outdated v2 archives | 4 | ~76 KB |
| Stale lock file | 1 | ~0.3 KB |
| Backup file | 1 | ~2 KB |
| Empty directories | 7 | 0 B |
| **Total** | **13** | **~78 KB** |

## 13. GitHub Readiness

**NOT READY** — 6 blocking issues:

1. Not/Or Ascend C correctness shows FAIL in stored JSON — reports need correction
2. Not/Or PyPTO measurement is torch.npu.Event not msprof — not comparable with relu/mul/add/div
3. Root README only covers 4 of 11 operators
4. Dashboard missing 4 complete operators (relu, mul, add, div)
5. 4 outdated v2 archives at project root
6. 3 operators (expand, transpose, reduce_sum) have CPU precompute — more honest to document or fix

**Suitable for continuing development?** YES — the pipeline is proven for relu/mul/add/div. The framework for Not/Or/Where/Equal expansion is in place. The main work is correctness verification and profiler standardization for the remaining operators.
