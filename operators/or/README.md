# Or Operator

## Implementation Status: COMPLETE

| Implementation | Correctness | B1 Latency |
|----------------|-------------|------------|
| torch | PASS | 256.3 us |
| ascendc | PASS | 6.5 us |
| pypto | PASS | 148.8 us |
## Kernel Details

torch: kernel=['?'], B1=256.3 us

ascendc: kernel=['or_kernel'], B1=6.5 us

pypto: kernel=['?'], B1=148.8 us
