# Cube Operator Template v2 — Project-Level Documentation

## Overview

Cube Operator Template v2 provides the standard directory structure, configuration files, and implementation patterns for Cube-class operators. It is designed for operators that use the AIC/Cube MMAD unit (vs. Vector/AIV SIMD).

## Applicable Operators

- **MatMul** (M×K @ K×N)
- **BatchMatMul** (batched variant)
- **Linear** (with optional bias)
- **QK** (attention score computation)
- **PV** (attention value projection)
- **FFN** (up/gate/down projections)
- **Gemm** (general matrix multiply)

## Key Differences from Vector Template v1

| Aspect | Vector v1 | Cube v2 |
|--------|-----------|---------|
| Compute Unit | AIV (Vector SIMD) | AIC (Cube MMAD) |
| Memory Hierarchy | GM → UB → Vector → GM | GM → L1 → L0A/L0B → MMAD → L0C → Fixpipe → GM |
| API | `AscendC::Add`, `DataCopyPad` | `MatmulImpl`, `Mmad`, `LoadData`, `Fixpipe` |
| Tiling | Element-based | M/N/K block-based |
| Performance Driver | Memory bandwidth | Cube compute throughput |

## Pipeline

```
GM A (ND/NZ) → L1 (DataCopy GM→L1)
                                   ↓
                            L0A (LoadData L1→L0A)
                                   ↓
GM B (ND/NZ) → L1 (DataCopy GM→L1)
                                   ↓
                            L0B (LoadData L1→L0B)
                                   ↓
                            Cube MMAD (Mmad/L0C)
                                   ↓
                            Fixpipe / DataCopy (L0C→GM)
                                   ↓
                            GM Y (ND/NZ)
```

## Tiling Parameters

| Field | Description |
|-------|-------------|
| M, N, K | Full matrix dimensions |
| baseM, baseN, baseK | Per-core tile size |
| depthA1, depthB1 | Double buffer depth for L1 |
| stepM, stepN | Iteration step size |
| batchM, batchN | Batch dimensions |
| BatchNum | Number of batch iterations |

## Data Format

- Input: FP16 ND (RowMajor)
- Cube internally may use NZ (fractal) format
- Output: FP16 ND

## Reuse Guide

### For BatchMatMul
- Same M,N,K as MatMul
- Set `batchM`/`batchN` to batch count
- Use `IterateBatch` or loop over batch

### For Linear
- M = batch × seq_len
- N = hidden_out
- K = hidden_in
- Add bias with `SetBias`

### For QK
- M = seq_len_q, N = seq_len_k, K = head_dim
- Follow with softmax (not in template)

### For PV
- M = seq_len_q, N = head_dim_v, K = seq_len_v
- Preceded by softmax output

### For FFN
- Three independent MatMuls:
  - up: [B,S,H] @ [H,4H]
  - gate: [B,S,H] @ [H,4H]
  - down: [B,S,4H] @ [4H,H]
- Each uses the same Cube pipeline
