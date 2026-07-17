# Release Changelog

## v1.2-rc2 (2026-07-17)

### PyPTO Unblocked — RC-2

| Operator | RC-1 Status | RC-2 Status | Root Cause | Workaround |
|----------|-------------|-------------|-----------|------------|
| **matmul** | BLOCKED_BACKEND | **UNBLOCKED** | Cube tiling FC4000 invalid tile values | `set_cube_tile_shapes([16,32],[16,32],[16,32])` |
| **div** | BLOCKED_BACKEND (P0) | **UNBLOCKED** | tile_shape(1024,2048) too large | Changed to (128,1024) |
| **where** | BLOCKED_BACKEND | **UNBLOCKED** | uint8 condition → ExpandFunction bug | Convert uint8→bool in wrapper |
| **transpose** | BLOCKED_BACKEND | **UNBLOCKED** | tile_shape(128,1024) too large | Changed to (64,256) |
| **equal** | BLOCKED_BACKEND | **UNBLOCKED** | DT_FP16 output + ta>64 for BOOL | DT_BOOL output + ta≤64 |

### Ascend C Perf — RC-2

| Operator | Change | Improvement |
|----------|--------|-------------|
| **transpose** | Tile size 32×32 + double buffering | ~13-18% across all batches |

### Phase 1 Completeness

| Area | Status |
|------|--------|
| Batch scaling audit | ✅ All 12 operators classified PLAUSIBLE_PARALLEL_SCALING |
| Parser traceability | ⚠️ 9 bugs identified (needs per-iteration stats and provenance) |
| Skill Trace | ✅ All 24 route-instances documented (LEGACY_UNVERIFIED_SKILL_USAGE) |
| SHA256SUMS | ✅ All 12 operators have valid SHA256SUMS (8 were fixed) |

### Known Limitations (RC-2)

| Operator | Route | Severity | Description |
|----------|-------|:--------:|-------------|
| or | PyPTO | P1 | Uses bitwise_or, no logical_or API |
| reduce_sum | all | P1 | FP16 accum precision > atol=0.01 |
| matmul | PyPTO | P2 | max_abs 0.015-0.031 (FP16 accum) |
| expand | PyPTO | P2 | AICPU dispatch ~3000 us |
| add | PyPTO | P2 | Correctness B=2..64 not persisted |

## v1.1 (2026-07-16)

### Status Changes

| Operator | v1.0 Status | v1.1 Status | Reason |
|----------|-------------|-------------|--------|
| **not** | COMPLETE_WITH_LIMITATION | **COMPLETE** | Ascend C correctness fixed (was FAIL/script bug, now 42/42 PASS) |
| **expand** | PARTIAL | **COMPLETE_WITH_LIMITATION** | Full correctness + msprof profiling completed |
| **transpose** | PARTIAL | **COMPLETE_WITH_LIMITATION** | Full correctness + msprof profiling for Torch+AscendC |
| **reduce_sum** | PARTIAL | **COMPLETE_WITH_LIMITATION** | Full validation completed; FP16 accum precision documented |

### P0 Fixes

1. **Not Ascend C** — Root cause: old correctness.py used wrong filename pattern (`reference_b{b}_bool.bin`). Current script iterates over 6 boundary cases per batch. All 42 cases PASS bitwise.
2. **Or Ascend C** — Same root cause as Not (old script bug). Current script has 7 variants × 7 batches = 49 cases. All 49 PASS bitwise.

### Or PyPTO Investigation

`pypto.logical_or` does NOT exist as an API. Only `pypto.bitwise_or` exists. For uint8 bool inputs (0/1 only), bitwise_or produces the same result as logical_or. Documented as a backend limitation.

### New Validations

| Route | Expand | Transpose | ReduceSum |
|-------|--------|-----------|-----------|
| Torch correctness | PASS B=1..64 bitwise | PASS B=1..64 bitwise | 62/70 PASS (7 NaN expected) |
| Ascend C correctness | PASS B=1..64 bitwise | PASS B=1..64 bitwise | 21/70 PASS (FP16 accum limit) |
| PyPTO correctness | PASS B=1..64 | Large BLOCKED_BACKEND | 21/70 PASS (FP16 accum limit) |
| Torch msprof | 13.0 us B=1 | 14.1 us B=1 | 16.4 us B=1 |
| Ascend C msprof | 15.0 us B=1 | 106.2 us B=1 | 14.4 us B=1 |
| PyPTO msprof | 3084 us AICPU dispatch | N/A (BLOCKED) | N/A |

### Corrections to Prior Reports

1. **Not/Or Ascend C correctness FAIL** → PASS (all cases bitwise)
2. **Expand/Transpose/ReduceSum HOST_PRECOMPUTE_FALLBACK** → TRUE_DEVICE_IMPLEMENTATION (confirmed by both source audit and NPU correctness)
3. **Expand/Transpose/ReduceSum PARTIAL** → COMPLETE_WITH_LIMITATION (validation completed)

### Remaining P0

- Div PyPTO broadcast backend blocker (no change)
