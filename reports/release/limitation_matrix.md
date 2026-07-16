# Known Limitations — Cannbot v1.1

## P0 (Blockers)

| Operator | Route | Issue |
|----------|-------|-------|
| div | PyPTO | Broadcast Div fails at backend CompileFunction |

## P1 (Should Fix)

| Operator | Route | Issue |
|----------|-------|-------|
| or | PyPTO | Uses bitwise_or (no logical_or API). Correct for 0/1 bool. |
| transpose | PyPTO | Large [256,384] BLOCKED_BACKEND at CompileFunction |
| equal | PyPTO | BLOCKED_BACKEND_EQUAL |
| where | PyPTO | BLOCKED_BACKEND_WHERE_SELECT |
| reduce_sum | All | FP16 accum precision exceeds atol=0.01 (max_abs~0.03 for random_finite) |

## P2 (Can Do Later)

| Operator | Route | Issue |
|----------|-------|-------|
| expand | PyPTO | Per-row AICPU dispatch ~3000 us — not compute kernel |
| add | PyPTO | Correctness B=2..64 not persisted to JSON |
| equal/not/or/where | All | No msprof profiling — Event-based only |
