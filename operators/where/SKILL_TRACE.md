# SKILL_TRACE: where

## Overview
- Operator: `where`
- Ascend C Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`
- PyPTO Classification: `NON_COMPLIANT`

## Ascend C Skill Usage
Standard kernel development pattern:
- Kernel source: `ascendc/src/where_kernel.asc`
- Host: `ascendc/src/where_host.asc`
- Tiling: `ascendc/src/where_tiling.h`
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
- `pypto/golden/where_golden.py`
- `pypto/src/where_impl.py`
- `pypto/tests/test_where.py`, `test_where_minimal.py`, `test_where_halfcond.py`, `test_where_halfcond2.py`

BLOCKED_BACKEND.
