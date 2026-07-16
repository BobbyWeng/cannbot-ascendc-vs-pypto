# Repair Backlog

## P0 — Must Fix (6 items)

| ID | Operator | Route | Issue | Risk |
|----|----------|-------|-------|------|
| P0-001 | not | Ascend C | Correctness script uses wrong filename pattern → all FAIL | low |
| P0-002 | or | Ascend C | Same correctness script bug | low |
| P0-003 | not | All | Reports claim PASS but stored correctness JSON shows FAIL | medium |
| P0-004 | or | All | Same report contradiction | medium |
| P0-005 | add | PyPTO | final_comparison.json reports AICPU init (~3ms) as primary compute instead of MIX_AIC (~136us) | low |

**Action required**: Fix correctness scripts, re-run on HW, correct final reports.

## P1 — Should Fix (12 items)

| ID | Operator | Route | Issue | Risk |
|----|----------|-------|-------|------|
| P1-001 | or | PyPTO | Uses bitwise_or instead of logical_or | medium |
| P1-002 | equal/not/or/where | Profiler | No msprof profiling; all use torch.npu.Event/aclrtEvent | low |
| P1-003 | expand | Ascend C | Broadcast expansion on CPU, not device | medium |
| P1-004 | transpose | Ascend C | Transpose on CPU, not device | medium |
| P1-005 | reduce_sum | Ascend C | FP32 pre-reduce on CPU | medium |
| P1-006 | div | Torch | Correctness B=4+ SKIP (files now exist) | low |
| P1-007 | div | Ascend C | Per-batch profiler only has B=32 | low |
| P1-008 | all | Docs | Root README only covers 4/11 operators | low |
| P1-009 | all | Dashboard | Missing relu/mul/add/div | low |
| P1-010 | relu/mul/add | Archive | No archives exist | low |
| P1-011 | div | Archive | v2 at root; no v4 in archives/ | low |
| P1-012 | div | PyPTO | Missing DIAGNOSTIC_REPORT.md | low |

## P2 — Can Do Later (5 items)

| ID | Operator | Route | Issue | Risk |
|----|----------|-------|-------|------|
| P2-001 | all | Correctness | ULP measurement missing | low |
| P2-002 | all | SHA256 | Path format not standardized | low |
| P2-003 | equal/not/or/where | SHA256 | Empty SHA256SUMS files | low |
| P2-004 | all | Cleanup | Delete outdated v2 archives (~76 KB) | low |
| P2-005 | expand/transpose | PyPTO | pyto/ directory should be pypto/ | low |
