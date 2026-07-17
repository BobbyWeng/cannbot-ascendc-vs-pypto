# MatMul Multi-Core Batch Scheduling Audit

## Current Implementation Summary

### Source Files
- `operators/matmul/ascendc/src/matmul_kernel.asc` — Cube kernel, processes 1 matrix per call
- `operators/matmul/ascendc/src/matmul_host.asc` — Host driver, serial matrix loop
- `operators/matmul/ascendc/src/matmul_tiling.h` — No custom tiling (uses `TCubeTiling` from `kernel_tiling/kernel_tiling.h`)
- `operators/matmul/ascendc/src/data_utils.h` — File I/O utilities

### Key Audit Findings

#### 1. Kernel Launch Pattern
| Parameter | Current Value |
|-----------|---------------|
| Kernels per logical call (B=1) | **12** (1 per matrix, host-serialized) |
| Kernels per logical call (B=32) | **384** (1 per matrix, host-serialized) |
| blockDim in kernel launch | **1** (single AICore) |
| blockDimUser (cmd arg, unused in launch) | 20 |
| Total AICores on device | **20** (per `platform_ascendc->GetCoreNumAic()`) |

**Critical finding**: Current implementation launches 1 kernel per matrix with blockDim=1. Each kernel call processes exactly 1 matrix on 1 AICore. All matrix processing is serialized through host-side loop (`matmul_host.asc:161-169`).

#### 2. Memory Addressing
```cpp
// matmul_host.asc:64-69 — Host computes total memory
uint32_t totalMatrices = batch * HEADS;  // B × 12

// matmul_kernel.asc:28-30 — Kernel computes per-matrix offset
aGlobal.SetGlobalBuffer(aBase + matIdx * aStride, M * K);  // aStride = M*K = 65536
bGlobal.SetGlobalBuffer(bBase + matIdx * bStride, K * N);  // bStride = K*N = 8192
cGlobal.SetGlobalBuffer(cBase + matIdx * cStride, M * N);  // cStride = M*N = 8192
```
- Offsets are correct for ND-layout contiguous [B×12, M, K] A, [B×12, K, N] B, [B×12, M, N] C
- Stride = M×K = 256×256 = 65536 (A), K×N = 256×32 = 8192 (B), M×N = 256×32 = 8192 (C)

#### 3. Tiling Configuration
```cpp
// matmul_host.asc:93
tilingApi.SetDim(1);  // Single-dimension tiling — no batch awareness
tilingApi.SetOrgShape(M, N, K);  // 256, 32, 256 — per-matrix shape only
tilingApi.SetShape(M, N, K);     // Same
```
- Tiling is **not batch-aware**: `SetDim(1)` configures for single-core single-matrix
- The tiling is identical for all matrices — correct since all matrices are same shape
- No batch/head dimension in tiling

#### 4. Data Format
- A: GM, ND, FP16
- B: GM, ND, FP16
- C: GM, ND, FP32 (accumulation in FP32, output stored as FP32)
- Host converts FP32→FP16 for storage (`matmul_host.asc:221-227`)

#### 5. Profiler Measurement

##### Ascend C Parsed (B=1)
| Metric | Value |
|--------|-------|
| Kernel count (total) | 8400 (across all iterations) |
| Logical calls | 100 |
| Kernels/logical call | **84** (7 per matrix? No — 12 matrices × 7? Actually parsing may include warmup) |
| all_device_kernels_us_per_call | 298.4 µs |
| primary_compute_kernel_us | 10.36 µs (per single matmul_kernel) |
| Mean kernel time | 3.55 µs |

##### Ascend C Parsed (B=32)
| Metric | Value |
|--------|-------|
| Kernel count (total) | 268800 |
| Logical calls | 100 |
| Kernels/logical call | **2688** |
| all_device_kernels_us_per_call | 9521.3 µs |
| primary_compute_kernel_us | 10.48 µs |
| Mean kernel time | 3.54 µs |

##### Torch Parsed (B=1)
| Metric | Value |
|--------|-------|
| Kernels/logical call | **7** |
| all_device_kernels_us_per_call | 38.0 µs |
| primary_compute_kernel_us | 12.22 µs |

##### Torch Parsed (B=32)
| Metric | Value |
|--------|-------|
| Kernels/logical call | **7** |
| all_device_kernels_us_per_call | 224.6 µs |
| primary_compute_kernel_us | 63.3 µs |

**Interpretation**: Torch's `BatchMatMulV2` launches ~7 kernels per logical call (likely 1 compute + 6 auxiliary for format conversion, workspace, etc.). Our Ascend C launches 12 to 384 separate compute kernels — each with its own kernel launch overhead (~3-4 µs per launch).

#### 6. TFLOPS Calculation
```cpp
// matmul_host.asc:239-240
double totalFlops = 2.0 * (double)totalMatrices * M * N * K;
double tflops = totalFlops / res.median_us / 1e6;
```
- Per-matrix FLOPs: 2 × 256 × 32 × 256 = 4,194,304
- B=1 total: 12 × 4,194,304 = 50,331,648
- B=32 total: 384 × 4,194,304 = 1,610,612,736
- Current TFLOPS B=1: 50,331,648 / (12×10.36) / 1e6 = 0.405 TFLOPS (incorrect — uses single kernel time, not total)
- Current TFLOPS B=32: 1,610,612,736 / 9521.3 / 1e6 = 0.169 TFLOPS

#### 7. Defects

| # | Severity | Issue |
|---|----------|-------|
| 1 | **CRITICAL** | Host-serialized matrix dispatch — 12 to 384 kernel launches per call |
| 2 | **HIGH** | blockDim=1 per kernel — 0 multi-core utilization |
| 3 | **HIGH** | Kernel launch overhead dominates: ~3.5 µs per kernel, total overhead = matrices × 3.5 µs |
| 4 | **HIGH** | B=32 has 2688 kernels/logical call — profiler counts through all |
| 5 | **MEDIUM** | TFLOPS calculation uses single-kernel latency, not call-level total |
| 6 | **MEDIUM** | No batch-unique validation — all matrices use same input pattern |
| 7 | **LOW** | blockDimUser=20 is misleading — never reaches kernel launch |
| 8 | **LOW** | FP16 output via host-side cast (FP32→FP16) adds host overhead |

### AICore Count
Current device (via `ascendcPlatform->GetCoreNumAic()`): **20 AICores**
- Architecture: dav-2201 (Ascend 910B)
- Available cores: 20 (one full device)

### totalMatrices Formula Confirmed
```cpp
totalMatrices = B * 12  // Correct per SPEC
```
- B=1: 12 matrices
- B=8: 96 matrices
- B=32: 384 matrices
- B=64: 768 matrices (new, added for Post-RC3)

### Batch/Head Stride Confirmed
- A stride per matrix: M × K = 256 × 256 = 65,536 elements = 128 KB
- B stride per matrix: K × N = 256 × 32 = 8,192 elements = 16 KB
- Y stride per matrix: M × N = 256 × 32 = 8,192 elements = 16 KB (FP32)
- Batch stride: 12 × matrix_stride
- Head stride: matrix_stride (contiguous)

### Output Coverage
- Current: All matrices written, but serialized through host loop
- No tail-matrix handling needed (exact division)
- Each core in new design should handle matricesPerCore = ceil(totalMatrices / blockDim)

## Recommendation
Replace host-serialized single-core-per-matrix with:
1. Single kernel launch per logical call
2. blockDim = 20 (all available AICores)
3. Each core processes ceil(totalMatrices / 20) matrices
4. Core-local addressing using blockIdx and matrix stride
5. Eliminate 11-383 unnecessary kernel launches
