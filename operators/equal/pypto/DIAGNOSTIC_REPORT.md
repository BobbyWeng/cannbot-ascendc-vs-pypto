# PyPTO Equal Blocker Diagnostic Report

## Status: BLOCKED_BACKEND_EQUAL

## Test Results
- **[1,32] FP16 → FP16 output**: Compiles, runs, but produces wrong results
  - Sum of all elements for identical inputs: ~0.0002 (expected 32.0)
  - Most outputs are 0 where 1.0 expected
- **[1,256,384] FP16 → FP16 output**: CompileFunction error at Expand pass

## Observations
1. `pypto.eq` produces wrong results even on the simplest case
2. The output values are not simple 0/1 — they seem to contain raw bit-packed mask values reinterpreted as FP16
3. Output dtype (FP16) doesn't match logical BOOL — but changing to uint8 may also fail
4. The CompileFunction error on full shapes is the same pattern as Div's broadcast issue

## Conclusion
`pypto.eq` on dav-2201 has a backend bug in how it lowers comparison results. The operator either:
- Wrongly outputs packed bitmask as FP16 values, or
- Has incorrect mask expansion logic

This is NOT fixable from user code. Requires PyPTO framework fix.
