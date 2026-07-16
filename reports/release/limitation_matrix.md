# Known Limitations — Cannbot v1.0

Generated from `reports/release/current_release.json`.

## P0 (Blockers)

| Operator | Route | Issue | Blocker Type |
|----------|-------|-------|-------------|
| div | PyPTO | Broadcast Div [B,12,256,256]/[B,12,256,1] fails at backend CompileFunction | BLOCKED_BACKEND |
| not | Ascend C | Correctness FAIL — all batches missing reference_bool.bin (script filename bug) | UNVERIFIED_RESULT |
| or | Ascend C | Correctness FAIL — all batches missing reference_bool.bin (script filename bug) | UNVERIFIED_RESULT |
| or | PyPTO | Uses bitwise_or instead of logical_or — only correct for 0/1 inputs | FUNCTIONAL_ISSUE |

## P1 (Should Fix)

| Operator | Route | Issue | Blocker Type |
|----------|-------|-------|-------------|
| add | PyPTO | Correctness B=2..64 not persisted (only B=1 saved) | MISSING_EVIDENCE |
| not | All | No msprof profiling — Event-based only, NOT comparable with arithmetic | INCOMPARABLE_METHODOLOGY |
| or | All | No msprof profiling — Event-based only, NOT comparable with arithmetic | INCOMPARABLE_METHODOLOGY |
| equal | All | No msprof profiling — Event-based only, NOT comparable with arithmetic | INCOMPARABLE_METHODOLOGY |
| where | All | No msprof profiling — Event-based only, NOT comparable with arithmetic | INCOMPARABLE_METHODOLOGY |
| expand | All | No correctness/profiler run on HW. Kernel is TRUE_DEVICE but unverified. | MISSING_HARDWARE_VALIDATION |
| transpose | All | No correctness/profiler run on HW. Kernel is TRUE_DEVICE but unverified. | MISSING_HARDWARE_VALIDATION |
| reduce_sum | All | No correctness/profiler run on HW. Kernel is TRUE_DEVICE but unverified. | MISSING_HARDWARE_VALIDATION |

## P2 (Can Do Later)

| Operator | Route | Issue | Blocker Type |
|----------|-------|-------|-------------|
| transpose | PyPTO | Large shape [B,256,384] BLOCKED_BACKEND at CompileFunction | BLOCKED_BACKEND |
| equal | PyPTO | BLOCKED_BACKEND_EQUAL | BLOCKED_BACKEND |
| where | PyPTO | BLOCKED_BACKEND_WHERE_SELECT | BLOCKED_BACKEND |
| div | All | No B=64 profiler data; per-batch profiler only for B=32 | MISSING_EVIDENCE |
