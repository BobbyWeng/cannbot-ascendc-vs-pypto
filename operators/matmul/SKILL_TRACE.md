# SKILL_TRACE: matmul

## Overview
- Operator: `matmul`
- Ascend C Classification: `LEGACY_UNVERIFIED_SKILL_USAGE`
- PyPTO Classification: `NON_COMPLIANT`

## Ascend C Skill Usage
Standard kernel development pattern:
- Kernel source: `ascendc/src/matmul_kernel.asc`
- Host: `ascendc/src/matmul_host.asc`
- Tiling: `ascendc/src/matmul_tiling.h`

Skills identified:
- `ascendc-kernel-develop-workflow`
- `ascendc-api-best-practices`
- `ascendc-tiling-design`

## PyPTO Skill Usage
`.orchestrator_state.json` missing at operator root — `NON_COMPLIANT`.

PyPTO artifacts use flat file layout (no SPEC/API_REPORT/DESIGN subdirs):
- `pypto/SPEC.md` ✓
- `pypto/API_REPORT.md` ✓
- `pypto/DESIGN.md` ✓
- `pypto/matmul_golden.py` ✓
- `pypto/matmul_impl.py` ✓
- `pypto/test_matmul.py` ✓

BLOCKED_BACKEND — matmul backend support not available.
