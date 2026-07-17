# SKILL_TRACE: equal

## Overview
- Operator: `equal`
- Ascend C Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`
- PyPTO Classification: `NON_COMPLIANT`

## Ascend C Skill Usage
Standard kernel development pattern:
- Kernel source: `ascendc/src/equal_kernel.asc`
- Host: `ascendc/src/equal_host.asc`
- Tiling: `ascendc/src/equal_tiling.h`
- Build: `ascendc/CMakeLists.txt`

Skills identified:
- `ascendc-kernel-develop-workflow`
- `ascendc-api-best-practices`
- `ascendc-tiling-design`

## PyPTO Skill Usage
`.orchestrator_state.json` missing — `NON_COMPLIANT`.

Artifacts present:
- `pypto/SPEC/SPEC.md`
- `pypto/API_REPORT/API_REPORT.md`
- `pypto/DESIGN/DESIGN.md`
- `pypto/golden/equal_golden.py`
- `pypto/src/equal_impl.py`
- `pypto/tests/test_equal.py`, `test_eq_minimal.py`, `test_eq_debug.py`

Skills identified by artifact evidence. BLOCKED_BACKEND.
