# Equal Operator

## Implementation Status: COMPLETE_WITH_LIMITATION

| Implementation | Correctness | B1 Latency |
|----------------|-------------|------------|
| torch | PASS | 12.2 us |
| ascendc | PASS | 41.8 us |
| pypto | BLOCKED_BACKEND_EQUAL | N/A |
## Kernel Details

torch: kernel=['?'], B1=12.2 us

ascendc: kernel=['equal_kernel'], B1=41.8 us

pypto: BLOCKED_BACKEND_EQUAL
