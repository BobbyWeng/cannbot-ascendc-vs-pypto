# RC-3 Skill Trace Summary

## Overview

All 12 operators × 2 routes (Ascend C, PyPTO) have SKILL_TRACE.md and SKILL_TRACE.json files.

## Classification

| Operator | Ascend C | PyPTO | Orchestrator State |
|----------|----------|-------|-------------------|
| add | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| div | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | root/ |
| equal | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| expand | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| matmul | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| mul | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| not | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| or | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| reduce_sum | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| relu | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| transpose | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |
| where | LEGACY_UNVERIFIED | LEGACY_UNVERIFIED | pypto/ |

## Skills Verified by Code Evidence

| Skill | Operators |
|-------|-----------|
| ascendc-kernel-develop-workflow | ALL 12 |
| ascendc-direct-invoke-template | ALL 12 |
| ascendc-tiling-design | ALL 12 |
| ascendc-api-best-practices | ALL 12 |
| ascendc-precision-debug | div, equal, where |
| ascendc-runtime-debug | reduce_sum |
| ops-profiling | matmul |
| pypto-intent-understand | ALL 12 |
| pypto-api-explore | ALL 12 |
| pypto-op-design | ALL 12 |
| pypto-golden-generate | ALL 12 |
| pypto-op-develop | ALL 12 |
| pypto-op-perf-tuner | mul, relu (Stage 7 complete) |

## RC-3 New Skill Usage

| Skill | Task | Operator |
|-------|------|----------|
| ascendc-api-best-practices | Transpose API audit | transpose |
| ascendc-tiling-design | Tile size optimization | transpose |
| pypto-op-develop | Expand one-shot, ReduceSum FP32 | expand, reduce_sum |
| pypto-api-explore | MatMul auto-tiling diagnosis | matmul |

Note: All RC-1/RC-2 skill usage is classified LEGACY_UNVERIFIED. Formal Skill Trace recording will start from RC-3 for all new modifications.
