# SKILL_TRACE: add

## Overview
- Operator: `add`
- Ascend C Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`
- PyPTO Classification: `NON_COMPLIANT`

## Ascend C Skill Usage
Code patterns match Cannbot skills. Standard kernel development pattern observed:
- Kernel source: `ascendc/src/add_kernel.asc`, `ascendc/src/add_host.asc`
- Tiling: `ascendc/src/add_tiling.h`
- Build: `ascendc/CMakeLists.txt`

Skills identified:
- `ascendc-kernel-develop-workflow` — confirmed by kernel/host/tiling/CMake patterns
- `ascendc-api-best-practices` — confirmed by kernel API usage
- `ascendc-tiling-design` — confirmed by tiling header

Formal trace was not recorded at development time.

## PyPTO Skill Usage
PyPTO artifacts exist but `.orchestrator_state.json` is missing, making this `NON_COMPLIANT`.

Artifacts present:
- `pypto/SPEC/SPEC.md`
- `pypto/API_REPORT/API_REPORT.md`
- `pypto/DESIGN/DESIGN.md`
- `pypto/golden/add_golden.py`
- `pypto/src/add_impl.py`
- `pypto/tests/test_add.py`

Skills identified (by artifact evidence):
- `pypto-intent-understand`
- `pypto-api-explore`
- `pypto-golden-generate`
- `pypto-op-design`
- `pypto-op-develop`

## Notes
- PyPTO implementation is BLOCKED_BACKEND — add_impl.py exists but backend support is incomplete
- .orchestrator_state.json missing at operator root, cannot confirm orchestration compliance
