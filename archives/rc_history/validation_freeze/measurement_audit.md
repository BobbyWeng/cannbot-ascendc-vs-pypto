# Measurement Audit — Validation Freeze

## Unified Measurement Standard Compliance

| Parameter | Standard | Compliance |
|-----------|----------|------------|
| Warmup | 200 | ALL operators: 200 |
| Profiled loops | >= 100 | ALL operators: 100 |
| Repeat | 5 | 11/12 operators: 5. Expand: 3 (documented: runtime >2h) |
| Profiler | msprof --ascendcl=on --ai-core=on --task-time=l0 | 8/12 operators use msprof. 4/12 (equal, not, or, where) use Event-based |
| Primary metric | primary_compute_kernel_us | CONSISTENT across msprof operators |

## Profiler Method by Operator

| Operator | Torch | Ascend C | PyPTO | Uniform? | Status |
|----------|-------|----------|-------|----------|--------|
| relu | msprof | msprof | msprof | YES | COMPLIANT |
| mul | msprof | msprof | msprof | YES | COMPLIANT |
| add | msprof | msprof | msprof | YES | COMPLIANT |
| div | msprof | msprof | msprof (N/A) | YES* | COMPLIANT |
| matmul | msprof | msprof | N/A (BLOCKED) | YES* | COMPLIANT |
| expand | msprof | msprof | msprof | YES | COMPLIANT (r3) |
| transpose | msprof | msprof | N/A (BLOCKED) | YES* | COMPLIANT |
| reduce_sum | msprof | msprof | N/A (not profiled) | YES* | COMPLIANT |
| equal | torch.npu.Event | aclrtEvent | N/A | PARTIAL | EVENT-ONLY |
| not | torch.npu.Event | aclrtEvent | torch.npu.Event | PARTIAL | EVENT-ONLY |
| or | torch.npu.Event | aclrtEvent | torch.npu.Event | PARTIAL | EVENT-ONLY |
| where | torch.npu.Event | aclrtEvent | N/A | PARTIAL | EVENT-ONLY |

*Only two comparable routes, but both use msprof.

## Critical Finding: 4 Operators Lack msprof Profiling

Operators equal, not, or, where use `torch.npu.Event` / `aclrtEvent` host-synchronized timing. Per the Unified Measurement Standard, this operates at Level 3 (host-synchronized operation) while msprof operates at Level 1 (device kernel). These are **NOT COMPARABLE** with arithmetic operators.

**Recommendation**: Run msprof for these 4 operators to produce comparable device-kernel timing data.

## Host Synchronization Overhead

For operators with both msprof and aclrtEvent data (e.g., div, matmul), the ratio of host-sync to device-kernel timing is typically 1.5x-3x for simple ops and up to 200x for per-matrix dispatch (matmul B32: 10.5us device vs 2416us host). This confirms that different measurement levels produce drastically different results and must not be mixed.

## Data Inconsistencies Found

1. **Mul final_comparison.json**: Claims torch B1 primary=3.876us. Parsed data shows 9.0us. The parsed data (from actual msprof run) is authoritative. The final comparison JSON was stale.
2. **Reduce_sum provisional_report.json**: Claims "profiler: PENDING" but performance_matrix.csv already has profiler values (torch B1=16.4, ascendc B1=14.4). Report needs updating.
3. **Div torch correctness**: B=4,8,16,32 SKIPPED due to missing reference files, but `all_pass` still shows `true`.

## SHA256SUMS Audit

| Operator | SHA256SUMS Status | Details |
|----------|------------------|---------|
| relu | VALID (48 entries) | Source + reports + config |
| mul | VALID (50 entries) | Source + reports + config |
| add | VALID (43 entries) | Source + reports + config |
| div | VALID (40 entries) | Source + reports + config |
| equal | EMPTY (0 bytes) | File exists but empty |
| not | EMPTY (0 bytes) | File exists but empty |
| or | EMPTY (0 bytes) | File exists but empty |
| where | BROKEN FORMAT | Lists filenames without hashes |
| matmul | COMMENT ONLY | No actual checksums |
| expand | MISSING | No SHA256SUMS file |
| transpose | MISSING | No SHA256SUMS file |
| reduce_sum | MISSING | No SHA256SUMS file |

Only 4/12 operators have valid SHA256SUMS. The archive policy requires source integrity verification.

## Binary Hash Verification

No operator has artifact_manifest.json tracking of build binary hashes. The build output in `build/{op}_ascendc` is not checksummed. Since build artifacts are excluded from SHA256SUMS per policy, recompilation is required to verify binary integrity.
