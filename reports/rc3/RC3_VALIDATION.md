# RC-3 Validation Report

## Summary

| Check | Status | Details |
|-------|--------|---------|
| Regression Test | ✅ 36/36 PASS | SHA256, build, profiler, parsed validation |
| SHA256 | ✅ 12/12 PASS | All operators verified |
| Correctness | ✅ ALL PASS | All working implementations verified |
| Parser Traceability | ✅ Complete | All parsed files have parser_version + profiler_type |
| Profiler | ✅ msprof | All 12 operators use msprof |
| Final Audit | ✅ Complete | 3 issues found and fixed |

## Per-Operator Validation

| Operator | Status | Correctness | Profiler | SHA256 | Skill Trace |
|----------|--------|------------|----------|--------|-------------|
| add | ✅ | Torch+AC+PyPTO | msprof 21 files | ✅ | LEGACY_UNVERIFIED |
| div | ✅ | Torch+AC+PyPTO | msprof 12 files | ✅ | LEGACY_UNVERIFIED |
| equal | ✅ | Torch+AC+PyPTO | msprof 21 files | ✅ | LEGACY_UNVERIFIED |
| expand | ✅ | Torch+AC+PyPTO | msprof 21 files | ✅ | LEGACY_UNVERIFIED |
| matmul | ✅ | Torch+AC+PyPTO | msprof 12 files | ✅ | LEGACY_UNVERIFIED |
| mul | ✅ | Torch+AC+PyPTO | msprof 21 files | ✅ | LEGACY_UNVERIFIED |
| not | ✅ | Torch+AC+PyPTO | msprof 21 files | ✅ | LEGACY_UNVERIFIED |
| or | ✅ | Torch+AC+PyPTO | msprof 21 files | ✅ | LEGACY_UNVERIFIED |
| reduce_sum | ✅ | Torch+AC+PyPTO | msprof 14 files | ✅ | LEGACY_UNVERIFIED |
| relu | ✅ | Torch+AC+PyPTO | msprof 21 files | ✅ | LEGACY_UNVERIFIED |
| transpose | ✅ | Torch+AC+PyPTO | msprof 15 files | ✅ | LEGACY_UNVERIFIED |
| where | ✅ | Torch+AC+PyPTO | msprof 14 files | ✅ | LEGACY_UNVERIFIED |

## Environment

| Component | Version |
|-----------|---------|
| Platform | Ascend 910B |
| CANN | 9.0.0 |
| Python | 3.11 |
| PyPTO | 0.2.0 |
| PyTorch | 2.7.1 |
| Profiler | msprof with --ascendcl=on --ai-core=on --task-time=l0 |
