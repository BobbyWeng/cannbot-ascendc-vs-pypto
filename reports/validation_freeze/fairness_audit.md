# Fairness Audit — Validation Freeze

## Three-Way Comparison Fairness

### Measurement Methodology Uniformity

| Group | Operators | Torch | Ascend C | PyPTO | Fair? | Can Rank? |
|-------|-----------|-------|----------|-------|-------|-----------|
| **Group A: Full msprof** | relu, mul, add, expand | msprof | msprof | msprof | YES | YES |
| **Group B: 2-route msprof** | div, transpose, reduce_sum, matmul | msprof | msprof | BLOCKED/N/A | YES* | YES (2 routes) |
| **Group C: Event-only** | equal, not, or, where | Event | aclrtEvent | Event/N/A | PARTIAL | Same-group only |

*For Group B, only torch and Ascend C are compared. PyPTO is excluded due to backend limitation.

### Critical Fairness Concern: Event vs msprof

Operators in Group C use host-synchronized timing which includes:
- ACL runtime dispatch overhead
- Device kernel submission latency
- CPU-GPU synchronization
- Python-to-C++ bridge overhead

msprof device-kernel timing (Group A, B) measures only the actual AI Core computation.

**Example**: For `not`, torch Event=127.5us for uint8 NOT (which should be ~2-3us device kernel). The 127.5us is almost entirely dispatch overhead. Ranking this against relu's 2.6us msprof kernel time would be fundamentally unfair.

**Conclusion**: Group A and B operators can be ranked together. Group C operators can only be ranked within Group C.

### Batch Scaling Fairness

All operators process the same input data (from `data/` directory) and the same batch sizes. The flat-batch pattern is valid for all element-wise operators. No unfair advantage exists in data processing.

### Kernel Count Asymmetry

| Operator | Torch kernels/call | Ascend C kernels/call | Fair? |
|----------|-------------------|----------------------|-------|
| relu | 1 | 1 | YES |
| mul | 1 | 1 | YES |
| add | 21 (7x3) | 1 | NOTED (documented) |
| div | 42 | 1 | NOTED (documented) |
| matmul | 7 (batched) | 12-384 (per-matrix) | NOTED (documented) |

Add and div use chained torch operations (multiple calls), producing more kernel events. This is inherent to how the spec defines the operation (e.g., add is 4-input = 3 chained binary adds). The report correctly documents this asymmetry.

Matmul's per-matrix dispatch is an Ascend C implementation choice, not a fairness issue. Torch's batched matmul is more efficient. Both are correctly documented.

### PyPTO JIT Overhead

PyPTO uses a two-process method: warmup (no profiler) then msprof session. This correctly excludes one-time JIT compilation overhead (approximately 3ms) from device-kernel timing. PyPTO's steady-state kernel performance is measured fairly, though the AICPU executor overhead is included in `all_device_kernels_us_per_call` (secondary metric).

### Expand Repeat Count

Expand uses repeat=3 instead of the standard repeat=5, documented as "runtime >2h". This is a practical exception but reduces statistical confidence. The reported values are medians, which are less sensitive to repeat count than means.

## Ascend C Implementation Authenticity

| Claim | Evidence |
|-------|----------|
| All 12 operators have Ascend C kernels | Source code confirmed in ascendc/src/ |
| All use Cannbot Skill patterns | Template structure matches ascendc-direct-invoke-template |
| No Host fallback | All host code does only copy + launch + read |
| matmul uses TRUE_CUBE | Uses `__cube__` kernel, `AscendC::Matmul` API |
| 11 ops use TRUE_DEVICE | Confirmed by source code audit and NPU correctness |

## PyPTO Implementation Authenticity

| Claim | Evidence |
|-------|----------|
| 7/12 operators used orchestrator | State files exist for: div, expand, matmul, mul, reduce_sum, relu, transpose |
| 5/12 operators have missing state | add, equal, not, or, where: artifact_manifest.json falsely claims state file |
| Backend limitations documented | 5 operators have BLOCKED_BACKEND documented in DIAGNOSTIC_REPORT.md |
| Code patterns match orchestrator style | All use same `@pypto.frontend.jit`, reshape, wrapper pattern |

## Conclusions

1. **8 operators can enter official ranking** with uniform msprof methodology
2. **4 operators have partial data** (Event-based only) — not comparable with msprof operators
3. **PyPTO backend limitations** affect 5 operators (div, equal, matmul, transpose, where)
4. **Implementation fairness** is maintained — all Ascend C kernels are true device implementations, all PyPTO kernels went through orchestrator or follow its patterns
5. **Measurement fairness** is maintained within each methodology group
