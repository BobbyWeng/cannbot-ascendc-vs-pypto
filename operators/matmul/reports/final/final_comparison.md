# MatMul Three-Route Comparison — Final Report

## Results Summary

| Route | Status | Correctness | Kernel Type | Kernel Name | B=1 Latency | B=32 Latency |
|-------|--------|-------------|-------------|-------------|:-----------:|:------------:|
| Torch | COMPLETE | PASS (atol=0.03125) | KERNEL_AIC_CUBE | aclnnMatmul_BatchMatMulNd_BatchMatMulV2 | ~us | ~us |
| Ascend C | COMPLETE ✅ | PASS (max_abs=0.015625) | KERNEL_AIC_CUBE (__cube__) | matmul_kernel | **75.7 us** (12 mats) | **2381 us** (384 mats) |
| PyPTO | BLOCKED_BACKEND | N/A | N/A | N/A | N/A | N/A |

## Correctness

All 6 batch sizes (B=1,2,4,8,16,32) pass. Ascend C Cube output vs CPU FP32→half golden:

| Batch | Matrices | max_abs | mean_abs | Status |
|:-----:|:--------:|:-------:|:--------:|:------:|
| 1 | 12 | 0.007812 | 0.000001 | PASS |
| 4 | 48 | 0.007812 | 0.000001 | PASS |
| 16 | 192 | 0.015625 | 0.000001 | PASS |
| 32 | 384 | 0.015625 | 0.000001 | PASS |

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
```

## Profiler Data

| Metric | Torch (aclnnMatmul) | Ascend C (matmul_kernel) |
|--------|:-------------------:|:------------------------:|
| Kernel type | KERNEL_AIC_CUBE | KERNEL_AIC_CUBE |
| Per-matrix latency | ~us | ~6.2 us |
| AIC cycles | 309846 (700 ops) | 17301 (single op) |
| Cube MAC ratio | 3.4% | 4.4% |
| Scalar ratio | 43.0% | 47.2% |
| AIV usage | 0% | 0% |

## Performance (Ascend C Cube per-matrix dispatch)

| Batch | Matrices | blockDim | Total (us) | Per-matrix (us) | TFLOPS |
|:-----:|:--------:|:--------:|:----------:|:----------------:|:------:|
| 1 | 12 | 12 | 75.7 | 6.31 | 0.665 |
| 2 | 24 | 12 | 151.5 | 6.31 | 0.665 |
| 4 | 48 | 16 | 297.4 | 6.20 | 0.677 |
| 8 | 96 | 20 | 596.0 | 6.21 | 0.676 |
| 16 | 192 | 20 | 1315.4 | 6.85 | 0.612 |
| 32 | 384 | 20 | 2380.9 | 6.20 | 0.677 |

~6.2 us per single [256×256] × [256×32] → [256×32] matrix via Cube MMAD.
