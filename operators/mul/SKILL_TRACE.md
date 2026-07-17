# SKILL_TRACE: mul

## Overview
- Operator: `mul`
- Ascend C Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`
- PyPTO Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`

## Ascend C Skill Usage
Standard kernel development pattern:
- Kernel source: `ascendc/src/mul_kernel.asc`
- Host: `ascendc/src/mul_host.asc`
- Tiling: `ascendc/src/mul_tiling.h`
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
- `pypto/golden/mul_golden.py`
- `pypto/src/mul_impl.py`
- `pypto/tests/test_mul.py`

Skills identified by artifact evidence. State file present but no formal trace recorded at development time.
