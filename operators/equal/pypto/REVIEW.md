# G13 Code Review: equal (PyPTO)

**Date**: 2026-07-23T09:15:49.398831+00:00
**Reviewer**: pypto-op-orchestrator (automated review)
**Score**: 91/100 — PASS

## 1. Code Summary
Element-wise equality comparison kernel for FP16 inputs with BOOL output using pypto.eq.

## 2. API Correctness
pypto.eq(x1, x2) correct for elementwise equality. DT_BOOL output matches expected type. Note: uses pypto.eq (not pypto.op.equal) — both should work.

## 3. Memory & Tile Configuration
- Tile: set_vec_tile_shapes(64, 1024) — smaller tile row, good for comparison ops.
- Reshape: Standard 2D reshape pattern — correct.
- Kernels per call: 1
- Kernel name: tilefwk_0_mix_aic

## 4. Workarounds
None needed.

## 5. Known Limitations
FP16 input only. Output is bool (uint8). NaN semantics: NaN != NaN (consistent with torch.eq).

## 6. Performance (PyPTO 0.2.0 + torch 2.8.0 + torch_npu 2.8.0.post2 + CANN 9.0.0)
- B1 primary median: 53.428us
- B64 primary median: 95.298us
- Correctness: B1-B64 bitwise PASS
- Raw profiling: /tmp/opencode/prof_results/equal/
- Parsed profiling: /tmp/opencode/parsed_results/equal_parsed.json

## 7. Gate Status
- G8 (Correctness): PASS
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS

## 8. Review Notes
Comparison ops have predictable scaling. Larger shapes show linear growth due to output tensor (bool) size.
