# Not Operator

## Implementation Status: COMPLETE

| Implementation | Correctness | B1 Latency | Profiler |
|----------------|-------------|------------|----------|
| torch | ✅ PASS | 127.5 us | torch.npu.Event |
| ascendc | ✅ PASS (42/42 cases, all B=1..64) | 6.4 us | aclrtEvent |
| pypto | ✅ PASS | 136.6 us | torch.npu.Event |

## Correction

Previous audit reported Ascend C correctness FAIL (missing reference_bool.bin). Root cause was an older correctness script that used wrong filename pattern. Current correctness.py iterates over all 6 boundary cases × 7 batches = 42 cases. All 42 PASS.

Profiler is torch.npu.Event/aclrtEvent — NOT comparable with msprof arithmetic data.
