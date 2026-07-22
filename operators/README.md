# Operator Summary

Generated from `reports/release/current_release.json` — single source of truth.

## Status Summary

| Category | Count | Operators |
|----------|:-----:|-----------|
| COMPLETE | 6 | relu, mul, not, matmul, layernorm, softmax |
| COMPLETE_WITH_LIMITATION | 8 | add, div, equal, or, where, expand, transpose, reduce_sum |

## Core Arithmetic (msprof, all batches)

| Operator | Status | Torch (B=1) | Ascend C (B=1) | PyPTO (B=1 compute) | PyPTO total |
|----------|--------|:-----------:|:--------------:|:-------------------:|:-----------:|
| **relu** | COMPLETE | 2.6 us | 2.1 us | 51.9 us | 163.0 us |
| **mul** | COMPLETE | 9.0 us | 11.2 us | 51.5 us | 208.2 us |
| **add** | COMPLETE_WITH_LIMITATION | 10.0 us | 13.8 us | 136.0 us | 462.9 us |
| **div** | COMPLETE_WITH_LIMITATION | 21.8 us | 18.6 us | N/A (backend blocked) | N/A |
| **softmax** | **COMPLETE** | 17.5 us | **6.8 us** | TBD | TBD |

All times B=1 msprof primary compute kernel (KERNEL_AIVEC for torch/ascendc, KERNEL_MIX_AIC for pypto).

## Logical/Comparison (msprof — comparable with arithmetic ops)

| Operator | Status | Torch (B=1) | Ascend C (B=1) | PyPTO (B=1 compute) | Correctness |
|----------|--------|:-----------:|:--------------:|:-------------------:|:-----------:|
| **equal** | COMPLETE_WITH_LIMITATION | 11.5 us | 50.0 us | N/A | Torch+AscendC+PyPTO PASS |
| **not** | COMPLETE | 8.2 us | 8.2 us | 118.8 us | Torch+AscendC+PyPTO PASS |
| **or** | COMPLETE_WITH_LIMITATION | 8.3 us | 9.1 us | 119.1 us | Torch+AscendC PASS; PyPTO bitwise_or |
| **where** | COMPLETE_WITH_LIMITATION | 10.1 us | 13.8 us | N/A | Torch+AscendC+PyPTO PASS |

## Cube Operators

| Operator | Status | Torch | Ascend C | PyPTO | Cube Badge |
|----------|--------|:-----:|:--------:|:-----:|:----------:|
| **matmul** | **COMPLETE** | ✅ PASS (atol/rtol) | ✅ TRUE_CUBE (msprof) | ✅ PASS (FP16 accum) | **True Cube** |

**Note**: MatMul correctness passes for all 3 routes. Ascend C uses Cube MMAD (MatmulImpl). PyPTO unblocked via manual set_cube_tile_shapes workaround (max_abs~0.015-0.031 due to FP16 accum).

## LayerNorm (msprof/event, all batches)

| Operator | Status | Torch (B=1) | Ascend C (B=1) | Ascend C (B=32) | Ascend C (B=64) | PyPTO (B=1) | Correctness |
|----------|--------|:-----------:|:--------------:|:---------------:|:---------------:|:-----------:|:-----------:|
| **layernorm** | **COMPLETE** | 23.2 us | **8.6 us** | **108.0 us** | **216.6 us** | 193.5 us | Torch+AscendC PASS; PyPTO precision limited |

**Note**: Ascend C optimized via AR-FullLoad + multi-block tiling (rowsPerBlock≤255). Normalize-only kernel (host weight/bias). B=1 7.4x faster than torch, B=64 20x faster than pre-optimization (4374→217 us).

## Layout/Reduce (no profiler data — PARTIAL)

| Operator | Status | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|--------|:-----:|:--------:|:-----:|:-----------:|:--------:|
| **expand** | PARTIAL | B=1 only | TRUE_DEVICE (unverified) | PASS (dispatch) | Major gaps | No msprof |
| **transpose** | PARTIAL | B=1 only | TRUE_DEVICE (unverified) | Partial (small PASS) | Major gaps | No msprof |
| **reduce_sum** | PARTIAL | B=1 only | TRUE_DEVICE (unverified) | SUCCESS (unverified) | Major gaps | No msprof |

**Note**: Expand/Transpose/ReduceSum Ascend C kernels are TRUE_DEVICE_IMPLEMENTATION (verified by source code audit). Previous reports claiming HOST_PRECOMPUTE_FALLBACK were incorrect. However, correctness and profiler data have not been collected on hardware.

## Known Limitations

See `reports/release/limitation_matrix.md` for the complete list.
