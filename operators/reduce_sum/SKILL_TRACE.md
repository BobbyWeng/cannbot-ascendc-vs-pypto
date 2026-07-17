# SKILL_TRACE: reduce_sum

## Overview
- Operator: `reduce_sum`
- Ascend C Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`
- PyPTO Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`

## Ascend C Skill Usage
Standard kernel development pattern:
- Kernel source: `ascendc/src/reduce_sum_kernel.asc`
- Host: `ascendc/src/reduce_sum_host.asc`
- Tiling: `ascendc/src/reduce_sum_tiling.h`
- Build: `ascendc/CMakeLists.txt`

Skills identified:
- `ascendc-kernel-develop-workflow`
- `ascendc-api-best-practices`
- `ascendc-tiling-design`

## PyPTO Skill Usage
`.orchestrator_state.json` EXISTS in pypto subdir — `LEGACY_UNVERIFIED_SKILL_USAGE`.

Artifacts present:
- `pypto/SPEC/SPEC.md`
- `pypto/API_REPORT/API_REPORT.md`
- `pypto/DESIGN/DESIGN.md`
- `pypto/golden/reduce_sum_golden.py`
- `pypto/src/reduce_sum_impl.py`
- `pypto/tests/test_reduce_sum.py`

Skills identified by artifact evidence. State file present but no formal trace recorded at development time.
