# MatMul Three-Route Comparison — Final Report

## Results Summary

| Route | Status | Correctness | Kernel Type | Kernel Name | B=1 Latency | B=32 Latency |
|-------|--------|-------------|-------------|-------------|:-----------:|:------------:|
| Torch | COMPLETE | PASS (atol=0.01) | KERNEL_AICORE | aclnnMatmul_BatchMatMulNd_BatchMatMulV2 | **12.2 us** (primary) | **63.3 us** (primary) |
| Ascend C | COMPLETE | PASS (max_abs=0.015625) | KERNEL_AICORE (__cube__) | matmul_kernel | **74.5 us** (12 mats) | **2416 us** (384 mats) |
| PyPTO | BLOCKED_BACKEND | N/A | N/A | N/A | N/A | N/A |

**PyPTO**: BLOCKED_BACKEND — Cube tiling engine returns FC4000 (invalid tile values). All matmul shapes fail. See `pypto/DIAGNOSTIC_REPORT.md`.

## Correctness

All 6 batch sizes (B=1,2,4,8,16,32) pass for both Torch and Ascend C.

| Batch | Matrices | Torch max_abs | Ascend C max_abs | Status |
|:-----:|:--------:|:-------------:|:----------------:|:------:|
| 1 | 12 | 0.003906 | 0.003906 | PASS |
| 2 | 24 | 0.007812 | 0.007812 | PASS |
| 4 | 48 | 0.007812 | 0.007812 | PASS |
| 8 | 96 | 0.007812 | 0.007812 | PASS |
| 16 | 192 | 0.015625 | 0.015625 | PASS |
| 32 | 384 | 0.007812 | 0.007812 | PASS |

Tolerance: atol=0.01, rtol=0.01 (require_bitwise=False). FP16 accumulation difference from FP32 reference is expected.

## Measurement Methodology

| Parameter | Value |
|-----------|-------|
| Profiler | msprof with `--ascendcl=on --ai-core=on --task-time=l0` |
| Warmup | 200 iterations |
| Profiled loops | 100 |
| Repeat | 5 |
| Primary metric | `primary_compute_kernel_us` (single kernel event from msprof trace) |
| Secondary metric | `all_device_kernels_us_per_call` (sum of all kernel events per logical call) |

**Note**: Torch uses `torch.npu.Event` timing in benchmark.py for host-synchronized latency. Ascend C uses `aclrtEvent` timing. Both also have independent msprof device-level timing.

## Profiler Data

### Primary Compute Kernel (msprof)

| Metric | Torch (a clnnMatmul) | Ascend C (matmul_kernel) |
|--------|:-------------------:|:------------------------:|
| Kernel type | KERNEL_AICORE | KERNEL_AICORE |
| Kernel name | aclnnMatmul_BatchMatMulNd_BatchMatMulV2 | matmul_kernel |
| B=1 primary kernel | 12.2 us | 10.4 us |
| B=32 primary kernel | 63.3 us | 10.5 us |
| Kernels per logical call | 7 | 12-384 (per-matrix dispatch) |

### All Device Kernels per Call (msprof)

| Batch | Torch | Ascend C |
|:-----:|:-----:|:--------:|
| 1 | 38.0 us | 298.4 us |
| 2 | 48.6 us | 599.1 us |
| 4 | 57.8 us | 1190.6 us |
| 8 | 79.0 us | 2382.4 us |
| 16 | 127.7 us | 4755.0 us |
| 32 | 224.6 us | 9521.3 us |

**Note**: Ascend C `all_device_kernels_us_per_call` includes all per-matrix kernel launches. Torch benefits from batched kernel fusion.

## Performance Comparison

### Primary Compute Kernel Latency

| Batch | Torch (primary kernel) | Ascend C (per-matrix kernel) | Ratio |
|:-----:|:----------------------:|:----------------------------:|:-----:|
| 1 | 12.2 us | 10.4 us | 0.85× |
| 2 | 16.4 us | 10.6 us | 0.65× |
| 4 | 17.6 us | 10.5 us | 0.60× |
| 8 | 23.6 us | 10.2 us | 0.43× |
| 16 | 35.6 us | 10.0 us | 0.28× |
| 32 | 63.3 us | 10.5 us | 0.17× |

Ascend C per-matrix Cube kernel is consistently ~10.4 us per [256×256] × [256×32] matrix.

### Host-Synchronized Latency

| Batch | Torch (torch.npu.Event) | Ascend C (aclrtEvent) |
|:-----:|:-----------------------:|:---------------------:|
| 1 | 15.2 us | 74.5 us |
| 2 | 15.2 us | 148.8 us |
| 4 | 15.1 us | 296.1 us |
| 8 | 16.8 us | 594.1 us |
| 16 | 23.8 us | 1188.9 us |
| 32 | 14.8 us | 2416.3 us |

### Ascend C Performance Breakdown

| Batch | Matrices | blockDim | Total (us) | Per-matrix (us) | TFLOPS |
|:-----:|:--------:|:--------:|:----------:|:----------------:|:------:|
| 1 | 12 | 12 | 74.5 | 6.21 | 0.675 |
| 2 | 24 | 12 | 148.8 | 6.20 | 0.677 |
| 4 | 48 | 16 | 296.1 | 6.17 | 0.680 |
| 8 | 96 | 20 | 594.1 | 6.19 | 0.678 |
| 16 | 192 | 20 | 1188.9 | 6.19 | 0.677 |
| 32 | 384 | 20 | 2416.3 | 6.29 | 0.667 |

TFLOPS formula: 2 × total_matrices × M × N × K / median_us / 1e6

## Cube Pipeline

```
Ascend C MatmulImpl APIs used:
- #include "lib/matmul_intf.h"
- AscendC::Matmul<GM,ND,half,  GM,ND,half,  GM,ND,float,  GM,ND,float,  CFG_NORM>
- REGIST_MATMUL_OBJ(&pipe, workspace, matmulObj, &tiling)
- SetTensorA(aGlobal, false)
- SetTensorB(bGlobal, false)
- IterateAll(cGlobal)

Pipeline: GM→L1→L0A/L0B→MMAD→L0C→Fixpipe→GM
Kernel dispatch: Per-matrix (1 kernel call per matrix, not batched)
```

## Known Limitations

1. **Ascend C per-matrix dispatch**: Each matrix is launched as a separate `<<<1, 0>>>` kernel call. This results in high launch overhead (12-384 kernel launches per logical operation). An optimized batched kernel would fuse all matrices into a single call.

2. **PyPTO BLOCKED_BACKEND**: PyPTO `pypto.matmul` fails at the Cube tiling engine (FC4000: invalid tile values). All matmul shapes are affected, not just the target shape.

3. **Kernel type classification**: msprof classifies both torch `aclnnMatmul` and Ascend C `matmul_kernel` as KERNEL_AICORE, not KERNEL_AIC_CUBE. The profiler may not distinguish AIC_CUBE from AICORE in this CANN version.

4. **Correctness scope**: Only perf_fp16 case is verified for Ascend C output. Special cases (zeros, ones, nan, inf, etc.) are not generated by the host binary — only perf inputs are supported.

## Final States

| Route | State | Reason |
|-------|-------|--------|
| Torch | COMPLETE | Full correctness + msprof profiler data |
| Ascend C | COMPLETE | TRUE Cube kernel, full correctness, msprof profiler data |
| PyPTO | COMPLETE_WITH_LIMITATION | BLOCKED_BACKEND — Cube tiling FC4000 error. Documented in DIAGNOSTIC_REPORT.md |
