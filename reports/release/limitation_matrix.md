# Known Limitations — Cannbot v1.2-rc2

## P0 (Blockers)
_None remaining — all previously blocked PyPTO routes were unblocked in RC-2._

## P1 (Should Fix)

| Operator | Route | Issue |
|----------|-------|-------|
| or | PyPTO | Uses bitwise_or (no logical_or API). Correct for 0/1 bool. |
| reduce_sum | All | FP16 accum precision exceeds atol=0.01 (max_abs~0.03 for random_finite) |

## P2 (Can Do Later)

| Operator | Route | Issue |
|----------|-------|-------|
| matmul | PyPTO | max_abs ~0.015-0.031 due to FP16 accumulation (not bitwise) |
| expand | PyPTO | Per-row AICPU dispatch ~3000 us — not compute kernel |
| add | PyPTO | Correctness B=2..64 not persisted to JSON |
| equal/not/or/where | All | No msprof profiling — Event-based only |

---

## RC-2 Resolved Limitations

The following 5 PyPTO backend blockages were resolved in RC-2:

| Operator | RC-1 Status | RC-2 Status | Root Cause | Workaround |
|----------|-------------|-------------|-----------|------------|
| div | P0 blocked | ✅ **UNBLOCKED** | tile_shape(1024,2048) too large + dtype mismatch | Changed to (128,1024). All 6 batches bitwise. |
| equal | P1 blocked | ✅ **UNBLOCKED** | DT_FP16 output instead of DT_BOOL + ta>64 | DT_BOOL output, ta≤64 tile shape. All 7 batches bitwise. |
| where | P1 blocked | ✅ **UNBLOCKED** | uint8 condition → backend ExpandFunction bug | Convert uint8→bool in wrapper. All 7 batches bitwise. |
| transpose | P1 blocked | ✅ **UNBLOCKED** | tile_shape(128,1024) too large | Changed to (64,256). All shapes up to 2048×2048 bitwise. |
| matmul | P1 blocked | ✅ **UNBLOCKED** | Cube tiling FC4000 invalid tile values | Manual set_cube_tile_shapes. FP16 accum max_abs~0.015-0.031. |
