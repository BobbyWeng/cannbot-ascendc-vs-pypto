# Operator Summary — v1.2-rc2

Generated from `reports/release/current_release.json` — single source of truth.

## Status Dashboard

| Operator | Final Status | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------------|-------|----------|-------|-------------|----------|
| relu | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full batch (7/7) | msprof |
| mul | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full batch (7/7) | msprof |
| not | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full batch (corrected) | Event |
| matmul | **COMPLETE** | PASS | TRUE_CUBE | **UNBLOCKED** ✅ | All 3 routes (6 batches) | msprof |
| add | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | PyPTO B=1 only | msprof |
| div | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | **UNBLOCKED** ✅ | All 3 routes (6 batches bitwise) | msprof |
| equal | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | **UNBLOCKED** ✅ | All 3 routes (7 batches bitwise) | Event |
| or | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | bitwise_or | AscendC corrected | Event |
| where | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | **UNBLOCKED** ✅ | All 3 routes (7 batches bitwise) | Event |
| expand | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | Full batch | msprof |
| transpose | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | **UNBLOCKED** ✅ | All 3 routes (7 batches bitwise) | msprof |
| reduce_sum | **COMPLETE_WITH_LIMITATION** | 62/70 | 21/70 (FP16) | 21/70 (FP16) | FP16 accum | msprof |

**Status counts**: 4 COMPLETE, 8 COMPLETE_WITH_LIMITATION, 0 PARTIAL, 0 INCOMPLETE

## Performance Summary (B=1 primary compute kernel, μs)

| Operator | Torch | Ascend C | PyPTO compute | Fastest |
|----------|:-----:|:--------:|:-------------:|:-------:|
| relu | 2.6 | 2.1 | 51.9 | Ascend C |
| mul | 9.0 | 11.2 | 51.5 | Torch |
| add | 10.0 | 13.8 | 136.0 | Torch |
| div | 21.8 | 18.6 | N/A | Ascend C |
| expand | 13.0 | 15.0 | 3084.5* | Torch |
| transpose | 14.1 | 106.2 | N/A | Torch |
| transpose (RC-2) | 14.1 | **~87-92** | N/A | Torch |
| reduce_sum | 16.4 | 14.4 | N/A | Ascend C |
| matmul | 12.2 | 10.4 | N/A | Ascend C |

*PyPTO expand = per-row AICPU dispatch, not compute kernel

## Key Changes in v1.2-rc2

### PyPTO Unblocked (5 operators)

| Operator | Root Cause | Workaround | Result |
|----------|-----------|------------|--------|
| **MatMul** | Cube tiling invalid tile values | Manual `set_cube_tile_shapes([16,32],[16,32],[16,32])` | All shapes compile and run; max_abs=0.015-0.031 (FP16 accum) |
| **Div** | tile_shape(1024,2048) too large + dtype mismatch in test | Changed to (128,1024) | All 6 batches bitwise (max_abs=0.0) |
| **Where** | uint8 condition → backend TiledWhereOperation ExpandFunction bug | Convert uint8→bool in wrapper (DT_BOOL condition) | All 7 batches bitwise |
| **Transpose** | tile_shape(128,1024) too large | Changed to (64,256) | All shapes up to 2048×2048 bitwise |
| **Equal** | (1) output was DT_FP16 instead of DT_BOOL, (2) BOOL output requires ta≤64 | Fixed dtype and tile shape | All 7 batches bitwise |

### Ascend C Perf (Phase 3)

| Operator | Change | Improvement |
|----------|--------|-------------|
| **Transpose** | Tile size 32×32 (up from 16×16) + double buffering | ~13-18% across all batches |

### Phase 1 Completeness

| Area | Status | Notes |
|------|--------|-------|
| Batch scaling audit | ✅ All 12 operators classified PLAUSIBLE_PARALLEL_SCALING |
| Parser traceability | ⚠️ 9 bugs identified | Needs per-iteration stats and provenance |
| Skill Trace | ✅ All 24 route-instances documented (LEGACY_UNVERIFIED_SKILL_USAGE) |
| SHA256SUMS | ✅ All 12 operators have valid SHA256SUMS (8 were fixed) |
