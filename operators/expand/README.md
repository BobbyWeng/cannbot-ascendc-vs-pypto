# Expand — Ascend C vs Torch vs PyPTO Comparison

## Operator
`Y[b,i,j] = X[b,i,0]` — broadcast expand last dim (1→384)

## Shapes
- X: `[B, 256, 1]`, FP16
- Y: `[B, 256, 384]`, FP16
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}

## Status: COMPLETE_WITH_LIMITATION

## Ascend C Implementation: TRUE_DEVICE_IMPLEMENTATION
GM[1] per row → GetValue → Duplicate[384] → GM[384]. Verified bitwise exact B=1..64.

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ✅ PASS | B=1..64 bitwise exact |
| Ascend C | ✅ PASS | B=1..64 bitwise exact (TRUE_DEVICE) |
| PyPTO | ✅ PASS | B=1..64 finite elements match |

## Performance (msprof, warmup=200, loops=100, repeat=3)

| B | Torch (AIVEC) | Ascend C (AIVEC) | PyPTO (AICPU) |
|---|:----:|:--------:|:------:|
| 1 | 13.0 us | 15.0 us | 3084.5 us |
| 2 | 14.5 us | 16.3 us | 2965.2 us |
| 4 | 13.4 us | 19.8 us | 2951.0 us |
| 8 | 14.5 us | 29.9 us | 3323.4 us |
| 16 | 15.0 us | 52.1 us | 3069.5 us |
| 32 | 16.2 us | 93.9 us | 3486.6 us |
| 64 | 17.5 us | 177.9 us | 2796.3 us |

## Known Issues
1. PyPTO uses per-row AICPU dispatch (~3000 us per call) — NOT a compute kernel. Not comparable with Torch/Ascend C.
2. Repeat reduced from 5 to 3 to stay under 2-hour runtime.
