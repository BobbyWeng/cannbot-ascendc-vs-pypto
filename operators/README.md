# Operator Summary

Generated from `reports/release/current_release.json` — single source of truth.

## Status Summary

| Category | Count | Operators |
|----------|:-----:|-----------|
| COMPLETE | 2 | relu, mul |
| COMPLETE_WITH_LIMITATION | 6 | add, div, equal, not, or, where |
| PARTIAL | 3 | expand, transpose, reduce_sum |

## Core Arithmetic (msprof, all batches)

| Operator | Status | Torch (B=1) | Ascend C (B=1) | PyPTO (B=1 compute) | PyPTO total |
|----------|--------|:-----------:|:--------------:|:-------------------:|:-----------:|
| **relu** | COMPLETE | 2.6 us | 2.1 us | 51.9 us | 163.0 us |
| **mul** | COMPLETE | 9.0 us | 11.2 us | 51.5 us | 208.2 us |
| **add** | COMPLETE_WITH_LIMITATION | 10.0 us | 13.8 us | 136.0 us | 462.9 us |
| **div** | COMPLETE_WITH_LIMITATION | 21.8 us | 18.6 us | N/A (backend blocked) | N/A |

All times B=1 msprof primary compute kernel (KERNEL_AIVEC for torch/ascendc, KERNEL_MIX_AIC for pypto).

## Logical/Comparison (torch.npu.Event/aclrtEvent — NOT comparable with msprof)

| Operator | Status | Torch (B=1) | Ascend C (B=1) | PyPTO | Correctness |
|----------|--------|:-----------:|:--------------:|:-----:|:-----------:|
| **equal** | COMPLETE_WITH_LIMITATION | 12.2 us | 41.8 us | BLOCKED_BACKEND | Torch+AscendC PASS |
| **not** | COMPLETE_WITH_LIMITATION | 127.5 us | 6.4 us | 136.6 us | AscendC FAIL (script bug); PyPTO UNVERIFIED |
| **or** | COMPLETE_WITH_LIMITATION | 256.3 us | 6.5 us | 148.8 us | AscendC FAIL (script bug); PyPTO bitwise_or |
| **where** | COMPLETE_WITH_LIMITATION | 131.9 us | 238.6 us | BLOCKED_BACKEND | Torch+AscendC PASS |

## Layout/Reduce (no profiler data — PARTIAL)

| Operator | Status | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|--------|:-----:|:--------:|:-----:|:-----------:|:--------:|
| **expand** | PARTIAL | B=1 only | TRUE_DEVICE (unverified) | PASS (dispatch) | Major gaps | No msprof |
| **transpose** | PARTIAL | B=1 only | TRUE_DEVICE (unverified) | Partial (small PASS) | Major gaps | No msprof |
| **reduce_sum** | PARTIAL | B=1 only | TRUE_DEVICE (unverified) | SUCCESS (unverified) | Major gaps | No msprof |

**Note**: Expand/Transpose/ReduceSum Ascend C kernels are TRUE_DEVICE_IMPLEMENTATION (verified by source code audit). Previous reports claiming HOST_PRECOMPUTE_FALLBACK were incorrect. However, correctness and profiler data have not been collected on hardware.

## Known Limitations

See `reports/release/limitation_matrix.md` for the complete list.
