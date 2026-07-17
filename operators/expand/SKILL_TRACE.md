# SKILL_TRACE: expand

## Overview
- Operator: `expand`
- Ascend C Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`
- PyPTO Classification: `NON_COMPLIANT`

## Ascend C Skill Usage
Standard kernel development pattern:
- Kernel source: `ascendc/src/expand_kernel.asc`
- Host: `ascendc/src/expand_host.asc`
- Tiling: `ascendc/src/expand_tiling.h`
- Build: `ascendc/CMakeLists.txt`

Skills identified:
- `ascendc-kernel-develop-workflow`
- `ascendc-api-best-practices`
- `ascendc-tiling-design`

## PyPTO Skill Usage
`.orchestrator_state.json` missing at operator root — `NON_COMPLIANT`.

PyPTO artifacts present but SPEC/API_REPORT/DESIGN subdirectories are EMPTY:
- `pypto/SPEC/` → empty
- `pypto/API_REPORT/` → empty
- `pypto/DESIGN/` → empty
- `pypto/golden/expand_golden.py` ✓
- `pypto/src/expand_impl.py` ✓
- `pypto/tests/test_expand.py` ✓

BLOCKED_BACKEND — backend support not available for this operator.
