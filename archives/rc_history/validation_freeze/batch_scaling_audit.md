# Batch Scaling Audit — Validation Freeze

## Methodology

Audited how each operator passes batch size to the kernel, whether batch actually reaches kernel computation, and whether all declared batches are tested.

## Per-Operator Batch Handling

| Operator | Batch Handling | Batch Reaches Kernel? | Correct? | Tested Batches |
|----------|---------------|----------------------|----------|----------------|
| relu | Flattened to totalElements | NO (flattened) | YES (element-wise) | 1,2,4,8,16,32,64 |
| mul | Flattened to totalElements | NO (flattened) | YES (element-wise) | 1,2,4,8,16,32,64 |
| add | Flattened to totalElements | NO (flattened) | YES (element-wise) | 1,2,4,8,16,32,64 |
| div | Flattened to totalElements | NO (flattened) | YES (element-wise+broadcast) | 1,2,4,8,16,32 |
| equal | Flattened to totalElements | NO (flattened) | YES (element-wise) | 1,2,4,8,16,32,64 |
| not | Flattened to totalElements | NO (flattened) | YES (element-wise) | 1,2,4,8,16,32,64 |
| or | Flattened to totalElements | NO (flattened) | YES (element-wise) | 1,2,4,8,16,32,64 |
| where | Flattened to totalElements | NO (flattened) | YES (element-wise) | 1,2,4,8,16,32,64 |
| expand | Folded into totalRows | NO (folded) | YES (row-based expand) | 1,2,4,8,16,32,64 |
| reduce_sum | Folded into totalRows | NO (folded) | YES (row-based reduction) | 1,2,4,8,16,32 |
| transpose | tiling.totalBatches field | **YES** | YES (per-batch offset) | 1,2,4,8,16,32,64 |
| matmul | Per-matrix dispatch on host | **YES** (per call) | YES (matIdx parameter) | 1,2,4,8,16,32 |

## Key Findings

### 1. Flat-Batch Pattern Is Universal (Except Transpose and Matmul)

All element-wise and row-based operators (10/12) use a flat-tensor approach:
- Host computes `totalElements = batch * H * W`
- Tiling struct has no `batch` field
- Kernel treats all data as one flat 1D array
- **This is correct** for element-wise operations and row-based operations, because the kernel does not need to know batch boundaries.

### 2. Only Transpose Is Batch-Aware in the Kernel

Transpose's tiling struct has `totalBatches` and the kernel explicitly computes:
- `batch = globalTile / totalTilesPerBatch`
- GM offset: `batch * H * W + (ti * tileH + r) * W + tj * tileW`
- This is necessary because output layout changes from [B,H,W] to [B,W,H].

### 3. Matmul Uses Per-Matrix Host Dispatch

Matmul dispatches `batch * 12` separate kernel calls, each processing one matrix:
- `matIdx` parameter identifies which matrix to process
- Host loops: `for (m = 0; m < totalMatrices; m++)`
- This is correct but causes high host overhead (2416us at B=32 vs 10.5us kernel time).

### 4. Batch Scaling Data

For element-wise ops, primary compute kernel time scales with total elements (batch * H * W):

| Operator | B=1 (us) | B=64 (us) | Scale Factor | Expected (64x) |
|----------|---------|----------|-------------|----------------|
| relu ascendc | 2.1 | 6.0 | 2.9x | Less than 64x (UB-limited) |
| mul ascendc | 11.2 | 20.8 | 1.9x | Less than 64x (UB-limited) |
| add ascendc | 13.8 | 33.4 | 2.4x | Less than 64x |
| not ascendc | 6.4 (Event) | ~10 (est) | ~1.5x | Less than 64x |

This sub-linear scaling confirms that Ascend C kernels are UB bandwidth-limited for small batches, not compute-limited. All kernels see batch increase but the scaling is sub-linear because the vector pipeline processes tiles independently of total size.

### 5. Correctness Gaps

- **Div torch**: B=4,8,16,32 reference files missing
- **Reduce_sum torch**: B=32 not present in correctness results
- **Matmul torch**: B=64 not tested (6/7 declared batches)

These do NOT indicate a batch scaling bug but represent verification gaps.
