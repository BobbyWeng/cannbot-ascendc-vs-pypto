# 4-Input FP16 Add Comparison — Unified msprof
## Y = ((X1+X2)+X3)+X4, shape [B,256,384]
Generated: 2026-07-16
Profiler: msprof (warmup=200, loops=100, repeat=5)

## Correctness

| Implementation | Status |
|---------------|--------|
| Torch NPU | ✅ PASS (all 7 batches, bitwise) |
| Ascend C | ✅ PASS (all 7 batches, bitwise) |
| PyPTO | ✅ PASS (all 7 batches, bitwise) |

## Performance (primary compute kernel, µs)

| B | Torch (KERNEL_AIVEC) | Ascend C (KERNEL_AIVEC) | PyPTO (KERNEL_MIX_AIC) | PyPTO total (incl. AICPU) | Fastest |
|---|:----:|:--------:|:------:|:--------:|:-------:|
| 1 | 10.04 | 13.78 | 136.03 | 462.9 | Torch |
| 2 | 11.08 | 19.26 | 139.345 | 471.404 | Torch |
| 4 | 10.86 | 17.68 | 144.892 | 485.146 | Torch |
| 8 | 10.42 | 14.94 | 156.495 | 528.198 | Torch |
| 16 | 12.68 | 22.54 | 176.376 | 569.455 | Torch |
| 32 | 20.56 | 33.381 | 210.826 | 645.057 | Torch |
| 64 | 27.541 | 54.601 | 260.94 | 778.684 | Torch |

## Kernel Type Breakdown (B=32)

### Torch
- Kernel types: ['KERNEL_AIVEC']
- Kernels per logical call: 21.0
- Kernel names: ['aclnnAdd_AddAiCore_Add']
- Primary compute: 20.56 µs
- All device kernels: 122.557 µs

### Ascendc
- Kernel types: ['KERNEL_AIVEC']
- Kernels per logical call: 7.0
- Kernel names: ['add_kernel']
- Primary compute: 33.381 µs
- All device kernels: 83.864 µs

### Pypto
- Kernel types: ['KERNEL_AICPU', 'KERNEL_MIX_AIC']
- Kernels per logical call: 9.01
- Kernel names: ['KERNEL_AICPU', 'PYPTO_add_binary_kernel']
- Primary compute: 3220.444 µs
- All device kernels: 645.057 µs
- AICPU executor: 434.231 µs

## Notes
- PyPTO uses two-process method: warmup (no profiler) → msprof session, JIT excluded
- Torch 3x torch.add produces ~7 kernel events per call (includes ACL overhead)
- Ascend C fused add: 1 kernel per call
- PyPTO 3x chained binary adds: 1 KERNEL_MIX_AIC + 2 KERNEL_AICPU per add = 9 events per call
- All measurements from msprof parsed data (not torch.npu.Event)
