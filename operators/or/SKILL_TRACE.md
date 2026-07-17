# SKILL_TRACE: or

## Overview
- Operator: `or`
- Ascend C Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`
- PyPTO Classification: `NON_COMPLIANT`

## Ascend C Skill Usage
Standard kernel development pattern:
- Kernel source: `ascendc/src/or_kernel.asc`
- Host: `ascendc/src/or_host.asc`
- Tiling: `ascendc/src/or_tiling.h`
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
- `pypto/golden/or_golden.py`
- `pypto/src/or_impl.py`
- `pypto/tests/test_or.py`

Skills identified by artifact evidence.
