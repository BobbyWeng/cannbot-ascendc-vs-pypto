# Operator Summary

Generated from `reports/release/current_release.json` — single source of truth.

## Status Dashboard

| Operator | Final Status | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------------|-------|----------|-------|-------------|----------|
| relu | **COMPLETE** | ✅ PASS | ✅ TRUE_DEVICE | ✅ SUCCESS | ✅ Full batch (7/7) | ✅ msprof |
| mul | **COMPLETE** | ✅ PASS | ✅ TRUE_DEVICE | ✅ SUCCESS | ✅ Full batch (7/7) | ✅ msprof |
| add | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ✅ SUCCESS | ⚠️ PyPTO B=1 only persisted | ✅ msprof |
| div | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ❌ BLOCKED_BACKEND | ✅ Torch+AscendC (B=1..32) | ✅ msprof |
| equal | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ❌ BLOCKED_BACKEND | ✅ Torch+AscendC | ⚠️ torch.npu.Event |
| not | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ✅ SUCCESS | ❌ AscendC FAIL (script bug) | ⚠️ torch.npu.Event |
| or | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ⚠️ bitwise_or | ❌ AscendC FAIL (script bug) | ⚠️ torch.npu.Event |
| where | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ❌ BLOCKED_BACKEND | ✅ Torch+AscendC | ⚠️ torch.npu.Event |
| expand | **PARTIAL** | ⚠️ B=1 only | ✅ TRUE_DEVICE (unverified) | ✅ PASS | ⚠️ Major gaps | ❌ No msprof |
| transpose | **PARTIAL** | ⚠️ B=1 only | ✅ TRUE_DEVICE (unverified) | ⚠️ Partial | ⚠️ Major gaps | ❌ No msprof |
| reduce_sum | **PARTIAL** | ⚠️ B=1 only | ✅ TRUE_DEVICE (unverified) | ✅ SUCCESS (unverified) | ⚠️ Major gaps | ❌ No msprof |

## Performance Summary (B=1 primary compute kernel, μs)

| Operator | Torch | Ascend C | PyPTO compute | Fastest |
|----------|:-----:|:--------:|:-------------:|:-------:|
| relu | 2.6 | 2.1 | 51.9 | Ascend C |
| mul | 9.0 | 11.2 | 51.5 | Torch |
| add | 10.0 | 13.8 | 136.0 | Torch |
| div | 21.8 | 18.6 | N/A | Ascend C |

## Important Corrections

1. **Expand/Transpose/ReduceSum Ascend C**: Source code confirmed as **TRUE_DEVICE_IMPLEMENTATION**. Duplicate, tile-transpose, ReduceSum Level 2 kernels. Previous HOST_PRECOMPUTE_FALLBACK claim was incorrect.
2. **Not/Or Ascend C**: Correctness FAIL due to missing reference_bool.bin files (script bug). Kernel may be correct.
3. **All Event-based profilers**: Not comparable with msprof arithmetic data.
