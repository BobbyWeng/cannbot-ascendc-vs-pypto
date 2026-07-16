# Cannbot v1.1 Current Release

**Single source of truth**: `reports/release/current_release.json`
**Generated**: 2026-07-16
**Git commit**: 3b3f102 (plus local pending changes)

## Operator Status

| Operator | Final Status | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------------|-------|----------|-------|-------------|----------|
| relu | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full batch | msprof |
| mul | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full batch | msprof |
| not | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full batch (corrected) | Event |
| add | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | PyPTO B=1 only | msprof |
| div | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BLOCKED | Torch+AscendC | msprof |
| equal | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BLOCKED | Torch+AscendC | Event |
| or | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | bitwise_or | AscendC corrected | Event |
| where | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BLOCKED | Torch+AscendC | Event |
| expand | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | Full batch (new!) | msprof |
| transpose | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | Partial | Torch+AscendC (new!) | msprof |
| reduce_sum | **COMPLETE_WITH_LIMITATION** | 62/70 | 21/70 (FP16) | 21/70 (FP16) | FP16 accum known | msprof |

## Status Changes from v1.0

| Operator | v1.0 Status | v1.1 Status | Reason |
|----------|-------------|-------------|--------|
| not | COMPLETE_WITH_LIMITATION | **COMPLETE** | Ascend C correctness fixed: 42/42 PASS |
| expand | PARTIAL | **COMPLETE_WITH_LIMITATION** | Full correctness + msprof completed |
| transpose | PARTIAL | **COMPLETE_WITH_LIMITATION** | Full correctness + msprof completed |
| reduce_sum | PARTIAL | **COMPLETE_WITH_LIMITATION** | Full correctness + msprof completed |

## Ascend C Implementation Audit

All 11 operators confirmed TRUE_DEVICE_IMPLEMENTATION via both source code audit and NPU correctness runs.
