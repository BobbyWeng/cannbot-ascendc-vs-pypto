# Ascend C MatMul — Audit Report

## Overview

The current Ascend C MatMul implementation in `operators/matmul/ascendc/` is audited against project standards. The README currently claims `TRUE_DEVICE_AIVEC (Vector path with FP32 accumulation, NOT Cube)` — this audit determines whether the implementation is a genuine Cube kernel or not.

## Audit Items

### 1. Kernel Type: TRUE Cube ✅

| Item | Status | Evidence |
|------|--------|----------|
| Kernel annotation | **`__cube__`** | `extern "C" __global__ __cube__ void matmul_kernel(...)` at `matmul_kernel.asc:60` |
| Cube API | **MatmulImpl high-level** | `AscendC::Matmul<GM,ND,half, GM,ND,half, GM,ND,float, GM,ND,float, CFG_NORM>` at `matmul_kernel.asc:38-44` |
| Pipeline registration | **REGIST_MATMUL_OBJ** | `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.matmulObj, &tiling)` at `matmul_kernel.asc:76` |
| IterateAll | **Cube MMAD** | `matmulObj.IterateAll(cGlobal)` at `matmul_kernel.asc:34` |
| No Vector path | **Confirmed** | No `__vector__` functions or Vector API calls |

**Verdict**: This is a **TRUE CUBE KERNEL** using high-level Cube MatMul API. The project README `operators/matmul/README.md` line 7 incorrectly states `TRUE_DEVICE_AIVEC` — this is outdated and must be corrected.

### 2. Host Fallback / ACLNN Wrapper ✅

| Item | Status | Evidence |
|------|--------|----------|
| ACLNN | **Not used** | No `aclnnMatmul*` API calls. Direct `<<<>>>` launch using ACL runtime. |
| Host precompute | **None** | Tiling uses `MultiCoreMatmulTiling` API; all compute is device-side. |
| Identity kernel | **Not used** | Kernel performs actual Cube MMAD computation. |

**Verdict**: No host fallback. This is a genuine device-side kernel.

### 3. BlockDim & Dispatch ⚠️

| Item | Status | Evidence |
|------|--------|----------|
| blockDim | **20 (configurable)** | Default 20, adjustable via CLI arg at `matmul_host.asc:48` |
| Multi-core distribution | **Manual** | `matricesPerCore = (totalMatrices + blockDim - 1) / blockDim` at `matmul_host.asc:76` |
| kernel launch | **1 kernel per matrix** | `for each matrix: matmul_kernel<<<1, 0, stream>>>` at `matmul_host.asc:166` |

**Issue**: The host launches **N separate kernel calls** (one per matrix) rather than a single batched kernel call. This means:
- For B=1 (12 matrices): 12 kernel launches
- For B=32 (384 matrices): 384 kernel launches
- Each launch has `<<<1, 0>>>` (blockDim=1 for Cube context)

**Impact**: High launch overhead. The profiler data will show one kernel event per matrix, inflating the kernel count by 12-384x compared to what a fused batched kernel could achieve.

### 4. Tiling Analysis ✅

| Item | Status | Evidence |
|------|--------|----------|
| Tiling API | **MultiCoreMatmulTiling** | `matmul_host.asc:90` — correct platform-aware tiling generation |
| Type configuration | **Correct** | A: GM/ND/DT_FLOAT16, B: GM/ND/DT_FLOAT16, C: GM/ND/DT_FLOAT at `matmul_host.asc:93-105` |
| Shape | **M=256, N=32, K=256** | Correct per-matrix dimensions |
| SetDim | **1** | `tilingApi.SetDim(1)` — single core per Cube call |

**Issue**: `SetDim(1)` limits each Cube call to a single AI Core. This is because each `matmul_kernel<<<1, 0>>>` handles only one matrix. A better design would use `SetDim(blockDim)` to allow the tiling engine to distribute work across multiple cores for a single kernel call.

### 5. Accumulation Dtype ⚠️

| Item | Value | Impact |
|------|-------|--------|
| MatmulType C | `float` (DT_FLOAT) | Cube MMAD output in FP32 |
| MatmulType Bias | `float` (DT_FLOAT) | Cube MMAD bias in FP32 |
| CFG_NORM | **FP16 accumulation** | Intermediate accumulation uses FP16 |

**Note**: `CFG_NORM` means the Cube MMAD uses FP16 for internal accumulation (L0C). The output is in FP32 via FixPipe. The kernel template's `MatmulType<GM, ND, float>` for C means the output is stored as FP32 in GM, but `CFG_NORM` limits accumulation to FP16 precision.

**Impact**: This is actually the **correct** configuration for a Cube MatMul with FP16 accumulation. The PR (PReLU/PostReLU) converts L0C→FP32 before storing to GM. No data wastage issue per se, but the final host code casts back to FP16 anyway (`matmul_host.asc:227`).

### 6. Pipeline & Double Buffer ✅

| Item | Status |
|------|--------|
| GM→L1→L0A/L0B | **Handled by MatmulImpl** (internal pipeline management) |
| MMAD→L0C | **Handled by IterateAll** |
| L0C→Fixpipe→GM | **Handled by MatmulImpl End()** |
| Double buffer | **Automatic via REGIST_MATMUL_OBJ** |

**Verdict**: The high-level MatMul API manages all pipelining and double buffering internally. No manual EnQue/DeQue needed.

### 7. LocalTensor Usage ✅

The high-level MatMul API abstracts away LocalTensor management. The kernel code uses `GlobalTensor` only — LocalTensor is managed internally by `REGIST_MATMUL_OBJ` and `IterateAll`.

### 8. DataCopy ✅

Data movement (GM→L1→L0A/L0B) is managed internally by the MatmulImpl framework. No explicit DataCopy calls needed. Correct.

### 9. Batch Addressing ✅

| Item | Status | Evidence |
|------|--------|----------|
| Stride calculation | **Correct** | `aStride = M * K`, `bStride = K * N`, `cStride = M * N` |
| Indexing | **Correct** | `aBase + matIdx * aStride` |

For batched MatMul with shape [B*12, M, K] flattening, each "matrix" occupies `M*K` elements. The matrices are laid out contiguously in memory: matrix 0 = batch 0 head 0, matrix 1 = batch 0 head 1, ..., matrix 11 = batch 0 head 11, matrix 12 = batch 1 head 0, etc.

**Verdict**: Batch addressing is correct for the [B,12,M,K]-contiguous layout.

### 10. Correctness Status ⚠️

| Item | Status | Evidence |
|------|--------|----------|
| Output BINs exist | **Yes** | B=1,2,4,8,16,32 have output BINs |
| Special-case outputs | **Missing** | Only `perf_fp16` case outputs exist (named `output_b1.bin` etc.) |
| Cube correct vs ref | **Claims PASS** | Final report says max_abs=0.015625 for all batches vs FP16 reference |

**Issue**: The correctness checker expects output files like `output_b1_perf_fp16.bin`, but the host program only writes `output_b1.bin` (without the `_perf_fp16` suffix). The correctness script at `data/generation_scripts/correctness.py:37` expects `output_b{batch}_{case}.bin` format. **The correctness script and host binary have a naming mismatch**.

**Also**: The data generation uses batch=32, but the experiment_config notes B=64 is excluded due to resource limits. The SPEC.yaml lists batches [1,2,4,8,16,32] — no B=64.

### 11. Profiler Data Status ✅

| Item | Status |
|------|--------|
| msprof raw data | **Exists for torch_b1 and ascendc_b1** |
| All batches | **Missing** for B=2,4,8,16,32 (both torch and ascendc) |

**Verdict**: Profiler data is incomplete. Only B=1 has raw msprof data.

### 12. Benchmark Configuration ⚠️

| Item | Current | Project Standard |
|------|---------|-----------------|
| Warmup | **100** (hardcoded) | **200** (project standard) |
| Loops | **1000** (default) | **100** (project standard) |
| Repeat | **10** (default) | **5** (project standard) |

**Verdict**: The host binary's default parameters do NOT match project standards. The `run_all.sh` script correctly passes project-standard values, but the binary's defaults are misaligned.

## Summary

### What Ascend C HAS completed:

1. ✓ True Cube MMAD kernel using native Ascend C MatmulImpl API
2. ✓ Correct tiling using MultiCoreMatmulTiling
3. ✓ Correct batch addressing for [B,12,256,256] × [B,12,256,32]
4. ✓ Works for all batches B=1,2,4,8,16,32
5. ✓ Output BINs exist and can be compared
6. ✓ Correct pipeline management (GM→L1→L0A/L0B→MMAD→L0C→Fixpipe→GM)
7. ✓ No host fallback, no ACLNN wrapper, no identity kernel

### What is MISSING or needs fixing:

1. **⬜ README.md is outdated** — claims `TRUE_DEVICE_AIVEC`, should be `TRUE_CUBE_IMPLEMENTATION`
2. **⬜ Per-matrix kernel dispatch** — launches one `<<<>>>` call per matrix (12-384 launches), extremely inefficient
3. **⬜ tiling SetDim(1)** — should use proper multi-core dispatch for each matrix
4. **⬜ Correctness script expects `output_b{batch}_{case}.bin`** but host only produces `output_b{batch}.bin`
5. **⬜ Host binary default params** — warmup=100 (should be 200), loops=1000 (should be 100), repeat=10 (should be 5)
6. **⬜ Profiler data incomplete** — only B=1 raw data exists
7. **⬜ Special-case correctness outputs** — not generated (only perf_fp16)
8. **⬜ The kernel uses `#include "lib/matmul_intf.h"`** but the CMakeLists.txt does not set up the include path for this header — could fail on different CANN installations

### Overall Status

| Criterion | Status |
|-----------|--------|
| True Cube Kernel | ✅ YES |
| no host fallback | ✅ PASS |
| no ACLNN wrapper | ✅ PASS |
| correctness complete | ⚠️ Script/host naming mismatch; only perf_fp16 outputs exist |
| profiler data complete | ❌ Only B=1 |
| benchmark params match standard | ❌ Binary defaults wrong (but run_all.sh overrides) |
| README accurate | ❌ Outdated |
| Efficient dispatch | ❌ Per-matrix launch |
