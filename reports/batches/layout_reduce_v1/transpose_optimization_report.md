# Transpose Optimization Report

## Candidate Summary

| Candidate | Approach | B=1 (us) | Speedup | Correctness | Status |
|-----------|----------|----------|---------|-------------|--------|
| Baseline | 16x16 tile, blockNum=1 | 1733 | 1.0x | PASS | Baseline |
| A | Transpose-96 API Scene 7 | 13865 | 0.12x | FAIL (zeros) | REJECTED |
| B | 16x16 tile, correct block distrib | 105.4 | 16.5x | PASS B=1..64 | **WINNER** |
| C | 32x32 tile, correct block distrib | 99.4 | 17.4x | PASS B=1 | Minor improvement |

## Winner: Candidate B (16x16 tile + correct block distribution)

### Root cause of original 1733 us
The original code had `if (blockNum > batch) blockNum = batch;` which limited blockNum=1 for B=1, regardless of the actual number of tiles. With 384 tiles (16×24 for [256,384]) and only 1 block, all work was serialized on a single core.

### Fix
Tile-based block count: `blockNum = min(blockDim, totalTiles)` where totalTiles = batch * (H/16) * (W/16). For B=1: 384 tiles, blockNum=20, giving proper multi-core distribution.

### Final Implementation
- **Method**: 16×16 tile with per-element swap (GetValue/SetValue)
- **Tile size**: 16×16 (256 elements per tile, optimal for UB)
- **Signal operations per tile**: 256 GetValue + 256 SetValue
- **Pipeline**: DataCopyPad row-by-row → element swap → DataCopyPad column-by-column

### Performance (B=1, 16x16 tile)

| Metric | Value |
|--------|-------|
| Baseline | 1733 us |
| Optimized | 105.4 us |
| Speedup | 16.5x |
| Torch B=1 | 23.5 us |
| Ratio vs Torch | 4.5x |

## Remaining Bottleneck

GetValue/SetValue per-element access. Future optimization directions:
1. Reduce tile count with larger tiles (32×32 marginal improvement)
2. Transpose-96 API Scene 7 on dav-2201 (CORRECTNESS_FAILED in testing)
3. Vector-level transpose via shuffle/interleave instructions
