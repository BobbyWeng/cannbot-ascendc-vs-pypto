# Equal Operator

## Implementation Status: COMPLETE_WITH_LIMITATION

| Implementation | Correctness | B1 Latency |
|----------------|-------------|------------|
| torch | PASS | 12.2 us |
| ascendc | PASS | 41.8 us |
| pypto | **UNBLOCKED RC-2** | N/A (not profiled) |
## Kernel Details

torch: kernel=['?'], B1=12.2 us

ascendc: kernel=['equal_kernel'], B1=41.8 us

pypto: UNBLOCKED RC-2 (DT_BOOL output + ta≤64)

## PyPTO RC-2 Fix
- **Root causes**: (1) output tensor was DT_FP16 instead of DT_BOOL (packed bitmask format), (2) BOOL output requires ta≤64 in tile shape.
- **Workaround**: Fixed output dtype to DT_BOOL. Reduced tile shape to respect ta≤64 constraint.
- **Result**: All 7 batches pass bitwise.
