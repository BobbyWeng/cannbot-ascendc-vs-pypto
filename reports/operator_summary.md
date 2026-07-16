# Operator Summary

Generated from `reports/release/current_release.json` — single source of truth.

## Status Dashboard

| Operator | Final Status | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------------|-------|----------|-------|-------------|----------|
| relu | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full batch (7/7) | msprof |
| mul | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full batch (7/7) | msprof |
| not | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full batch (corrected) | Event |
| add | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | PyPTO B=1 only | msprof |
| div | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BLOCKED | Torch+AscendC | msprof |
| equal | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BLOCKED | Torch+AscendC | Event |
| or | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | bitwise_or | AscendC corrected | Event |
| where | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BLOCKED | Torch+AscendC | Event |
| expand | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | Full batch (new!) | msprof |
| transpose | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | Partial | Torch+AscendC (new!) | msprof |
| reduce_sum | **COMPLETE_WITH_LIMITATION** | 62/70 | 21/70 (FP16) | 21/70 (FP16) | FP16 accum | msprof |

## Performance Summary (B=1 primary compute kernel, μs)

| Operator | Torch | Ascend C | PyPTO compute | Fastest |
|----------|:-----:|:--------:|:-------------:|:-------:|
| relu | 2.6 | 2.1 | 51.9 | Ascend C |
| mul | 9.0 | 11.2 | 51.5 | Torch |
| add | 10.0 | 13.8 | 136.0 | Torch |
| div | 21.8 | 18.6 | N/A | Ascend C |
| expand | 13.0 | 15.0 | 3084.5* | Torch |
| transpose | 14.1 | 106.2 | N/A | Torch |
| reduce_sum | 16.4 | 14.4 | N/A | Ascend C |

*PyPTO expand = per-row AICPU dispatch, not compute kernel

## Key Changes in v1.1

1. **Not**: Ascend C correctness FAIL→PASS (old script bug, 42/42 PASS with current script)
2. **Or**: Ascend C correctness FAIL→PASS (same script fix, 49/49 PASS)
3. **Expand**: PARTIAL→COMPLETE_WITH_LIMITATION (full correctness + msprof completed)
4. **Transpose**: PARTIAL→COMPLETE_WITH_LIMITATION (full correctness + msprof for Torch+AscendC)
5. **ReduceSum**: PARTIAL→COMPLETE_WITH_LIMITATION (full validation, FP16 accum documented)
