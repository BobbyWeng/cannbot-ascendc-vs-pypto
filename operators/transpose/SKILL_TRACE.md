# SKILL_TRACE: transpose

## Overview
- Operator: `transpose`
- Ascend C Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`
- PyPTO Classification: `NON_COMPLIANT`

## Ascend C Skill Usage
Standard kernel development pattern:
- Kernel source: `ascendc/src/transpose_kernel.asc`
- Host: `ascendc/src/transpose_host.asc`
- Tiling: `ascendc/src/transpose_tiling.h`
- Build: `ascendc/CMakeLists.txt`

Skills identified:
- `ascendc-kernel-develop-workflow`
- `ascendc-api-best-practices`
- `ascendc-tiling-design`

## PyPTO Skill Usage
`.orchestrator_state.json` missing at operator root — `NON_COMPLIANT`.
PyPTO SPEC/API_REPORT/DESIGN subdirectories are EMPTY:
- `pypto/SPEC/` → empty
- `pypto/API_REPORT/` → empty
- `pypto/DESIGN/` → empty
- `pypto/golden/transpose_golden.py` ✓
- `pypto/src/transpose_impl.py` ✓
- `pypto/tests/test_transpose.py` ✓

BLOCKED_BACKEND.
