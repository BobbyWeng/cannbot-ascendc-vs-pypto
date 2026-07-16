# Operator Summary

## Core Arithmetic — COMPLETE

| Operator | Status | Torch (msprof) | Ascend C (msprof) | PyPTO (msprof MIX_AIC) |
|----------|--------|:--------------:|:-----------------:|:----------------------:|
| relu | COMPLETE | 2.6 us | 2.1 us | 51.9 us |
| mul | COMPLETE | 9.0 us | 11.2 us | 51.5 us |
| add | COMPLETE | 10.0 us | 13.8 us | 136.0 us |
| div | COMPLETE_WITH_LIMITATION | 21.8 us | 18.6 us | BLOCKED_BACKEND |

## Logical/Comparison — COMPLETE_WITH_LIMITATION (no msprof)

| Operator | Status | Torch (Event) | Ascend C (aclrtEvent) | PyPTO |
|----------|--------|:------------:|:--------------------:|:-----:|
| equal | COMPLETE_WITH_LIMITATION | 12.2 us | 41.8 us | BLOCKED_BACKEND_EQUAL |
| where | COMPLETE_WITH_LIMITATION | 131.9 us | 238.6 us | BLOCKED_BACKEND_WHERE |

## Logical — REPORT_OUTDATED (correctness FAIL)

| Operator | Status | Torch (Event) | Ascend C (aclrtEvent) | PyPTO |
|----------|--------|:------------:|:--------------------:|:-----:|
| not | REPORT_OUTDATED | 127.5 us | 6.4 us | 136.6 us |
| or | REPORT_OUTDATED | 256.3 us | 6.5 us | 148.8 us |

Not/Or: Ascend C correctness shows all batches FAIL (script bug). Reports falsely claim PASS.

## Layout/Reduce — INCOMPLETE

| Operator | Status | Torch | Ascend C | PyPTO |
|----------|--------|:-----:|:--------:|:-----:|
| expand | INCOMPLETE | B=1 only | HOST_PRECOMPUTE | PARTIAL |
| transpose | INCOMPLETE | B=1 only | HOST_PRECOMPUTE | BLOCKED (large) |
| reduce_sum | INCOMPLETE | B=1 only | HOST_PRECOMPUTE | SUCCESS (no correctness) |

All times B=1. Arithmetic operators use msprof (primary compute kernel). Logical/comparison operators use torch.npu.Event/aclrtEvent (host-synchronized) — NOT comparable.
