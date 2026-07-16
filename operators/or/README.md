# Or Operator

## Implementation Status: COMPLETE_WITH_LIMITATION

| Implementation | Correctness | B1 Latency | Profiler |
|----------------|-------------|------------|----------|
| torch | ✅ PASS | 256.3 us | torch.npu.Event |
| ascendc | ✅ PASS (49/49 cases, all B=1..64) | 6.5 us | aclrtEvent |
| pypto | ✅ PASS (bitwise_or for uint8 0/1) | 148.8 us | torch.npu.Event |

## Correction

Previous audit reported Ascend C correctness FAIL. Root cause was older correctness script with wrong filename pattern. Current correctness.py iterates over all 7 variants × 7 batches = 49 cases. All 49 PASS.

### PyPTO Limitation

PyPTO uses `pypto.bitwise_or(x1, x2)` instead of a hypothetical `pypto.logical_or`. Since `pypto.logical_or` does NOT exist in the framework (only `logical_and`, `logical_not`, and `bitwise_or` exist), and the input dtype is uint8 with 0/1 values, `bitwise_or` produces the correct result for all bool inputs.

This is documented as a backend limitation: PyPTO lacks `logical_or` API. For general uint8 values beyond 0/1, bitwise_or would differ from logical_or.
