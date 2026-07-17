# Where Operator

## Implementation Status: COMPLETE_WITH_LIMITATION

| Implementation | Correctness | B1 Latency |
|----------------|-------------|------------|
| torch | PASS | 131.9 us |
| ascendc | PASS | 238.6 us |
| pypto | **UNBLOCKED RC-2** | N/A (not profiled) |
## Kernel Details

torch: kernel=['?'], B1=131.9 us

ascendc: kernel=['where_kernel'], B1=238.6 us

pypto: UNBLOCKED RC-2 (uint8→bool DT_BOOL workaround)

## PyPTO RC-2 Fix
- **Root cause**: uint8 condition tensor triggers backend TiledWhereOperation ExpandFunction bug.
- **Workaround**: Convert uint8 condition to DT_BOOL condition in wrapper before calling PyPTO where.
- **Result**: All 7 batches pass bitwise.
