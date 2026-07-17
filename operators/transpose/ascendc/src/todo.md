# Task: RC-3 Transpose Kernel Optimization

## Goal
Achieve 5-10% improvement over RC-2 baseline (32×32 tile, double buffer)

## RC-2 Baseline (host event timing, 1000 loops, 10 repeats)
| Batch | 32×32 (µs) |
|-------|-----------|
| 1     | 85.0      |
| 2     | 168.4     |
| 4     | 335.0     |
| 8     | 652.0     |
| 16    | 1286.0    |
| 32    | 2570.5    |
| 64    | 5139.3    |

## Approaches

### Approach 1: Larger tile sizes
- [x] Save RC-2 backup
- [x] 48×48 tile — mostly slower (B=1: +32%, only B=64: +1% better)
- [x] 64×64 tile — best for large batches: B=4: +2.8%, B=32: +1.6%, B=64: +2.9%
- [x] 128×128 tile — competitive for B≥16 but worse for small batches

### Approach 2: Batch-aware tile processing
- [x] Implemented — significantly slower (SetValue/GetValue triple loop overhead)

### Approach 3: Pipeline optimization (TQue depth=3)
- [x] TQue depth=3 — 2x slower (more UB allocation, not DMA-bound)

### Approach 4: Combined best
- [x] Best: 64×64 tile for B≥4; 32×32 for B=1,2

### Approach 5: DMA transpose
- [x] Direct column-by-column DMA — very slow (th*tw DataCopyPad calls)

## Progress
All approaches evaluated. Final: use 64×64 for large batches (up to +2.9%)
