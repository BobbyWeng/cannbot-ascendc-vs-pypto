# Expand Provisional Comparison Report

**Status: COMPLETE** — Full profiler data collected for all 3 implementations across all 7 batch sizes.

## Experiment Configuration
- Shape: [B, 256, 1] -> [B, 256, 384]
- Dtype: float16
- Warmup: 200, Loops: 100, Repeat: 3
- Profiler: msprof (ascendcl=on, ai-core=on, task-time=l0)

## Correctness (All Batches B=1,2,4,8,16,32,64)

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch | PASS (all B) | Bitwise exact (max_abs_diff=0.0) |
| Ascend C | PASS (all B) | Bitwise exact, device-side Duplicate kernel |
| PyPTO | PASS (all B) | Finite-element match (max_abs_diff=0.0), per-row JIT dispatch |

## Profiler — Primary Compute Kernel (us)

| Batch | Torch | Ascend C | PyPTO |
|-------|-------|----------|-------|
| 1 | 13.02 | 15.04 | 3084.48 |
| 2 | 14.48 | 16.32 | 2965.24 |
| 4 | 13.36 | 19.84 | 3172.16 |
| 8 | 14.38 | 27.72 | 3721.78 |
| 16 | 11.94 | 49.66 | 3082.56 |
| 32 | 14.42 | 91.8 | 2918.7 |
| 64 | 17.5 | 177.86 | 2796.34 |

## Profiler — All Device Kernels (us)

| Batch | Torch | Ascend C | PyPTO |
|-------|-------|----------|-------|
| 1 | 20.188 | 3048.827 | 39083.833 |
| 2 | 21.557 | 3140.123 | 76812.663 |
| 4 | 22.026 | 3501.027 | 155007.229 |
| 8 | 23.64 | 6275.942 | 311213.719 |
| 16 | 25.052 | 12284.831 | 625020.655 |
| 32 | 33.485 | 23963.338 | 1265597.514 |
| 64 | 47.357 | 48298.528 | 2507501.139 |

## Profiler — Kernel Count

| Batch | Torch | Ascend C | PyPTO |
|-------|-------|----------|-------|
| 1 | 500 | 28192 | 79105 |
| 2 | 500 | 28192 | 158209 |
| 4 | 500 | 28192 | 316417 |
| 8 | 500 | 28192 | 632833 |
| 16 | 500 | 28192 | 1265665 |
| 32 | 500 | 28192 | 2531329 |
| 64 | 500 | 28192 | 5062657 |

## PyPTO Implementation Detail

- JIT kernel: 1D per-row 
- Each batch dispatches B × 256 kernel calls per iteration
  - B=1: 256 calls; B=64: 16384 calls
- Primary compute type: KERNEL_AICPU (PyPTO executor)
- This design is fundamentally different from native torch.expand and the Ascend C Duplicate kernel.

## Ascend C Implementation Detail

- Device-side Duplicate kernel (expand_kernel)
- 20 blocks, row-based work distribution
- Primary compute type: KERNEL_AIVEC

## Torch Implementation Detail

- Native torch_npu expand
- Primary compute type: KERNEL_AIVEC
- All batches show ~500 total kernel events (100 loops × 5) with ~3 kernels per logical call

## Notes

- PyPTO kernel count grows linearly with batch size (B*256*3 = per-row expand_clone + AICPU overhead)
- Ascend C and Torch use native device operations with much lower overhead
- Parse divides by 100 loops; actual profiled iterations = repeat × loops = 300