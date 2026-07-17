# MatMul Multi-Core Batch Audit

## Current Implementation

### Scheduling
- **Old**: Host-serialized per-matrix kernel launch (B×12 launches per logical call)
- **New**: Single kernel launch, multi-core dispatch via `GetBlockIdx()`/`GetBlockNum()`

### Kernel Count
| Batch | Old compute kernels/call | New compute kernels/call | Reduction |
|-------|------------------------|------------------------|-----------|
| 1     | 12 (serial)            | 1 (parallel)           | 12×       |
| 2     | 24                     | 1                      | 24×       |
| 4     | 48                     | 1                      | 48×       |
| 8     | 96                     | 1                      | 96×       |
| 16    | 192                    | 1                      | 192×      |
| 32    | 384                    | 1                      | 384×      |

### blockDim
- **Old**: 1 per kernel launch
- **New**: min(totalMatrices, 20) — uses all available AICores

### Matrices per Core
| Batch | totalMatrices | blockDim | matricesPerCore |
|-------|-------------|---------|----------------|
| 1     | 12          | 12      | 1              |
| 2     | 24          | 20      | 2              |
| 4     | 48          | 20      | 3              |
| 8     | 96          | 20      | 5              |
| 16    | 192         | 20      | 10             |
| 32    | 384         | 20      | 20             |
| 64    | 768         | 20      | 39 (works)     |

### Waves
- B=1: 12 cores active (1 matrix each), 8 idle → 1 wave
- B=2: 20 cores, 4 get 2 matrices, 16 get 1 → 2 waves
- B=32: 20 cores, all get 20 matrices → 20 sequential waves per core

### N=32 Analysis
- Cube N direction utilization: **Low** (32 < 128 Cube native)
- ND→NZ conversion overhead: Internal to MatmulImpl
- Core load balance: Near-perfect (max 1 matrix imbalance)
- L0/L1 reuse: Per-matrix (each `IterateAll` uses full L0A/L0B/L0C)

## Device Details
- Architecture: dav-2201 (Ascend 910B)
- AICores: 20 per device
- Memory: 64 GB HBM
- CANN: 9.0.0

## Recommendations
1. For B≥2, all 20 cores are fully utilized
2. Consider NBuffer33 schedule for deeper pipeline on larger batches
3. No B=1 specialization needed (correctness maintained)
