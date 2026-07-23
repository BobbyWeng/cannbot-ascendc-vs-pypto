# G13 Code Review: reduce_sum (PyPTO)

**Date**: 2026-07-23T09:52:46.105537+00:00
**Score**: 80/100 — PASS_WITH_NOTES

## 1. Code Summary
FP32 accumulation reduction sum: FP16 input → cast to FP32 → pypto.op.sum(dim=-1) → cast to FP16.
FP32 kernel tile(64,384) for 384-element last-dim reduction.

## 2. API Correctness
pypto.op.sum supports FP32 only (not FP16). Cast → reduce → cast pattern is correct.
tile(64,384) sized for FP32 elements (4 bytes each).

## 3. Correctness (re-verified)
- B1-B64: PASS (atol=0.02, FP16 vs FP32-accumulated reference)
- Max observed diff: 0.016 (B=8, within FP16 rounding tolerance for 384-element sum)

## 4. Performance
- B1: MIX_AIC median 42.6us (kernels_per_call=2: 1 reduction + 1 cast)
- B64: MIX_AIC median 71.8us (64x rows, 1.68x kernel time)

## 5. Gate Status
- G8 (Correctness): PASS
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS_WITH_NOTES

## 6. Notes
- Not bitwise-perfect due to different FP32 accumulation order vs torch
- max_diff 0.016 within acceptable FP16 reduction tolerance
