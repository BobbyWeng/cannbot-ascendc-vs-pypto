# G13 Code Review: transpose (PyPTO)

**Date**: 2026-07-23T09:28:01.534850+00:00
**Score**: 82/100 — PASS

## 1. Code Summary
2D transpose via `pypto.op.transpose(x, 0, 1)`. Tile shape (64,256).
Per-batch JIT kernel loop with reshape → batch dispatch.

## 2. API Correctness
`pypto.op.transpose` is the correct PyPTO API for 2D transpose.
`set_vec_tile_shapes(64,256)` is mandatory before the op (JIT constraint).

## 3. Implementation Notes
- Wraps B×H×W in per-batch 2D slices → 1 kernel launch per batch element
- B=64 means 64 JIT kernel launches per logical call
- Tile(64,256) works; tile(128,512) ~9% faster

## 4. Performance
- Primary kernel (MIX_AIC): ~46us per 384×256 tile
- AICPU overhead: ~2×49us per kernel launch
- B=1: 1 kernel, B=64: 64 kernels (linear scaling)

## 5. Correctness (re-verified today)
- B1-B64: bitwise PASS
- Sentinel (fp16 max, min subnormal): PASS
- Batch-unique data: PASS
- Negative values: PASS

## 6. Gate Status
- G8 (Correctness): PASS
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS

## 7. Issues
- Per-batch loop adds Bx dispatch overhead (inherent to 2D-only API)
