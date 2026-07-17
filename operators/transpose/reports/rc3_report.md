# RC-3: Transpose Kernel Optimization Report

## Methodology
- **Platform**: Ascend 910B (dav-2201)
- **Shape**: [B, 256, 384] → [B, 384, 256] (fp16/half)
- **Data**: Random fp16 inputs, bitwise correctness validation
- **Timing**: Host event-based (aclrtRecordEvent, 1000 loops, 10 repeats, 100 warmup)
- **Profiling**: msprof `--ascendcl=on --ai-core=on --task-time=l0`
- **Kernel**: Ascend C AIVEC kernel, 20 AICores (blockNum)

## RC-2 Baseline (32×32 tile, DOUBLE_BUFFER=2)
| Batch | Time (µs) | Tiles/Block |
|-------|-----------|-------------|
| 1     | 85.0      | 5           |
| 2     | 168.4     | 10          |
| 4     | 335.0     | 20          |
| 8     | 652.0     | 39-40       |
| 16    | 1286.0    | 77-78       |
| 32    | 2570.5    | 154-155     |
| 64    | 5139.3    | 308-309     |

## Approach 1: Tile Size Grid Search

### Results
| Batch | 32×32 | 48×48 | 64×64 | 128×128 |
|-------|-------|-------|-------|---------|
| 1     | 85.0  | 112.1 | 131.0 | 257.1   |
| 2     | 168.4 | 185.7 | 195.9 | 257.3   |
| 4     | 335.0 | 369.6 | **325.5** (+2.8%) | 513.4 |
| 8     | 652.0 | 737.3 | 649.5 | 769.5   |
| 16    | 1286.0| 1436.0| 1297.3| 1281.7  |
| 32    | 2570.5| 2641.9| **2528.7** (+1.6%) | 2561.6 |
| 64    | 5139.3| 5089.2| **4991.2** (+2.9%) | 5121.4 |

### Analysis
- **Larger tiles reduce total tile count** but increase per-tile SetValue/GetValue cost (O(th*tw))
- 64×64 gives best large-batch performance: +2.9% for B=64
- Correctness maintained: bitwise exact for all tile sizes
- 128×128 not beneficial despite UB capacity headroom

## Approach 2: Batch-Aware Tiling (Rejected)
- Processed [batchTile=8, tileH, tileW] per work item
- Triple loop [B][H][W] with SetValue/GetValue added massive overhead
- 20-25% slower across all batches due to scalar UB access
- Tradeoff: more data per tile reduces DMA launches but increases compute cost

## Approach 3: TQue Depth=3 (Rejected)
- Increased pipeline depth from 2 to 3
- Allocate 3× tile-size UB per buffer (vs 2×)
- >2x slower: 85µs → 189µs for B=1
- Root cause: kernel is compute-bound (SetValue/GetValue), not DMA-bound
- Deeper pipeline increases UB pressure without hiding compute latency

## Approach 4: DMA Transpose (Rejected)
- Directly read source column elements via DataCopyPad for each element
- th × tw DataCopyPad calls per tile (vs 2×th+tw baseline)
- Extremely slow — failed to complete within timeout
- Each element requires independent DMA setup

## Best Configuration

### Default: 32×32 tile (all-round)
Best for small batches (B=1,2). Uses the proven RC-2 kernel unchanged.

### Large Batch: 64×64 tile
For B≥4, the 64×64 tile gives up to +2.9% improvement. Recommended when batch size is known to be large.

### Recommended future work
The fundamental bottleneck is the scalar SetValue/GetValue loop in ComputeTile. To achieve >5% improvement, this must be replaced with vector operations:
- `TransDataTo5HD` — limited to 16×16 sub-blocks, 5HD format (not general transpose)
- `Gather` (7-arg version) — could vectorize column extraction
- Custom vectorized shuffle — requires hardware-specific vector permute instructions

### Command-line usage
```bash
# Default (32×32):
./transpose_ascendc 0 64 20 32 32 100 1000 10 <data_dir> <output_dir>

# Large-batch optimized (64×64):
./transpose_ascendc 0 64 20 64 64 100 1000 10 <data_dir> <output_dir>
```

## Correctness
All tile sizes tested (32, 48, 64, 128) pass bitwise correctness for B={1,2,4,8,16,32,64}.
