# Not Operator

## Implementation Status: COMPLETE

| Implementation | Correctness | B1 Latency |
|----------------|-------------|------------|
| torch | PASS | 127.5 us |
| ascendc | PASS | 6.4 us |
| pypto | PASS | 136.6 us |
## Kernel Details

torch: kernel=['?'], B1=127.5 us

ascendc: kernel=['not_kernel'], B1=6.4 us

pypto: kernel=['?'], B1=136.6 us
