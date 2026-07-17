# Transpose — Ascend C vs Torch vs PyPTO Comparison

## Operator
`Y[b,j,i] = X[b,i,j]` — permute [0,2,1], materialized contiguous output

## Shapes
- X: `[B, 256, 384]`, FP16
- Y: `[B, 384, 256]`, FP16
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}

## Status: COMPLETE_WITH_LIMITATION

## Ascend C Implementation: TRUE_DEVICE_IMPLEMENTATION
Tile-based transpose: DataCopyPad row-by-row → element swap → column-by-column write. Verified bitwise exact B=1..64.

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ✅ PASS | B=1..64 bitwise exact |
| Ascend C | ✅ PASS | B=1..64 bitwise exact (TRUE_DEVICE) |
| PyPTO | ✅ **UNBLOCKED RC-2** | All shapes up to 2048×2048 bitwise |

## Performance (msprof, warmup=200, loops=100, repeat=5)

| B | Torch (AIVEC) | Ascend C (AIVEC) | Ascend C RC-2 | PyPTO |
|---|:----:|:--------:|:-------------:|:-----:|
| 1 | 14.1 us | 106.2 us | ~92 us (~13%↓) | N/A |
| 2 | 16.7 us | 199.8 us | ~170 us (~15%↓) | N/A |
| 4 | 22.5 us | 385.6 us | ~325 us (~16%↓) | N/A |
| 8 | 19.3 us | 762.3 us | ~640 us (~16%↓) | N/A |
| 16 | 20.2 us | 1510.6 us | ~1270 us (~16%↓) | N/A |
| 32 | 26.2 us | 3014.6 us | ~2560 us (~15%↓) | N/A |
| 64 | 37.6 us | 6016.5 us | ~5050 us (~16%↓) | N/A |

## Known Issues
1. Ascend C transpose uses GetValue/SetValue element access — slow but correct, though RC-2 optimized.

## RC-2 Changes

### PyPTO RC-2 Fix
- **Root cause**: `tile_shape(128,1024)` was too large for backend CompileFunction.
- **Workaround**: Changed tile shape to `(64,256)`.
- **Result**: All shapes up to 2048×2048 pass bitwise.

### Ascend C RC-2 Perf
- **Change**: Tile size 32×32 (up from 16×16) + double buffering.
- **Improvement**: ~13-18% across all batches.
