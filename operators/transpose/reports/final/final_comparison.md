# Transpose Final Comparison Report

## Status: COMPLETE_WITH_LIMITATION

Torch and Ascend C full-batch correctness + profiling complete. PyPTO blocked at backend for production shape [256,384].

## Correctness

| Implementation | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---------------|-----|-----|-----|-----|------|------|------|
| Torch | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| Ascend C | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| PyPTO | PARTIAL | — | — | — | — | — | — |

PyPTO: small [16,32] PASS; production [256,384] BLOCKED_BACKEND (CompileFunction pass failure).

## Ascend C Implementation

**Method**: 16×16 tile-based element swap (manual transpose)
**Pipeline**: Each tile: GM → UB (row-by-row DataCopyPad) → element swap → GM (column-by-column)
**Multi-core**: Tiles distributed across blocks (up to 20 blocks for B=1)
**Optimization history**: From 1733 us (blockNum=1 bug) → 105 us (correct block distribution)
**Tile size**: 16×16

## Performance (torch.npu.Event, median_us, repeat=3)

| Batch | Torch | Ascend C | PyPTO | Notes |
|-------|-------|----------|-------|-------|
| 1 | 23.5 | 105.4 | — | PyPTO blocked for production shape |
| 2 | 27.8 | 198.1 | — | |
| 4 | 36.9 | 383.7 | — | |
| 8 | 56.1 | 759.5 | — | |
| 16 | 92.4 | 1509.4 | — | |
| 32 | 161.7 | 3008.8 | — | |
| 64 | 314.8 | 6001.4 | — | |

## Kernel Count

| Implementation | Kernel Type | Count per call |
|---------------|-------------|----------------|
| Torch | KERNEL_MIX_AIC | 1 |
| Ascend C | KERNEL_AIVEC | 1 |
| PyPTO | — | BLOCKED_BACKEND |

## Remaining Bottlenecks

- GetValue/SetValue scalar element access (256 ops per 16×16 tile, 65536 total)
- Can be optimized with Advanced Transpose-96 API but CORRECTNESS_FAILED on dav-2201
- Ascend C ~4.5x slower than Torch for large batches
