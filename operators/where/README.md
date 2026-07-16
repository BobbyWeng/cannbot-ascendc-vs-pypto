# Where Operator

## Implementation Status: COMPLETE_WITH_LIMITATION

| Implementation | Correctness | B1 Latency |
|----------------|-------------|------------|
| torch | PASS | 131.9 us |
| ascendc | PASS | 238.6 us |
| pypto | BLOCKED_BACKEND_WHERE_SELECT | N/A |
## Kernel Details

torch: kernel=['?'], B1=131.9 us

ascendc: kernel=['where_kernel'], B1=238.6 us

pypto: BLOCKED_BACKEND_WHERE_SELECT
