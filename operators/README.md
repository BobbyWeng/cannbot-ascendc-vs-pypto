# Operator Summary

## Core Arithmetic (COMPLETE — msprof, all batches)

| Operator | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------|----------|-------|-------------|----------|
| **relu** | 2.6 us | 2.1 us | 51.9 us (compute) | PASS all B | msprof ✅ |
| **mul** | 9.0 us | 11.2 us | 51.5 us (compute) | PASS all B | msprof ✅ |
| **add** | 10.0 us | 13.8 us | 136.0 us (compute) | PASS all B | msprof ✅ |
| **div** | 21.8 us | 18.6 us | BLOCKED_BACKEND | PASS (Torch+AscendC) | msprof ⚠️ B=32 only |

All times are B=1 msprof primary compute kernel (KERNEL_AIVEC for torch/ascendc, KERNEL_MIX_AIC for pypto).

## Logical/Comparison (COMPLETE_WITH_LIMITATION — no msprof)

| Operator | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------|----------|-------|-------------|----------|
| **equal** | 12.2 us | 41.8 us | BLOCKED_BACKEND_EQUAL | PASS (Torch+AscendC) | torch.npu.Event ⚠️ |
| **where** | 131.9 us | 238.6 us | BLOCKED_BACKEND_WHERE | PASS (Torch+AscendC) | torch.npu.Event ⚠️ |

All times B=1 host-synchronized (torch.npu.Event/aclrtEvent). NOT comparable with arithmetic msprof data.

## Logical/Comparison (REPORT_OUTDATED — correctness FAIL or unverified)

| Operator | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------|----------|-------|-------------|----------|
| **not** | 127.5 us | 6.4 us | 136.6 us | FAIL (script bug) | torch.npu.Event ⚠️ |
| **or** | 256.3 us | 6.5 us | 148.8 us | FAIL (script bug) | torch.npu.Event ⚠️ |

Not/Or: Ascend C correctness JSON shows all batches FAIL. Reports falsely claim PASS. Must re-run with fixed script.
Or PyPTO: Uses bitwise_or instead of logical_or — only correct for 0/1 inputs.

## Layout/Reduce (INCOMPLETE — host precompute or incomplete)

| Operator | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------|----------|-------|-------------|----------|
| **expand** | B=1 only | HOST_PRECOMPUTE_FALLBACK | PARTIAL | NOT RUN | N/A |
| **transpose** | B=1 only | HOST_PRECOMPUTE_FALLBACK | BLOCKED_BACKEND (large) | NOT RUN | N/A |
| **reduce_sum** | B=1 only | HOST_PRECOMPUTE_FALLBACK | SUCCESS | NOT RUN | N/A |

Expand/Transpose/ReduceSum Ascend C kernels are identity copies with host precompute. Not true device-side implementations.
torch correctness only covers B=1.
