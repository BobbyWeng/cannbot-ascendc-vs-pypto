# G13 Code Review: mul (PyPTO)

**Date**: 2026-07-23T09:15:49.398831+00:00
**Reviewer**: pypto-op-orchestrator (automated review)
**Score**: 94/100 — PASS

## 1. Code Summary
Element-wise multiplication kernel for FP16 tensors using pypto.op.mul.

## 2. API Correctness
pypto.op.mul(x1, x2) correct for elementwise FP16 multiplication.

## 3. Memory & Tile Configuration
- Tile: set_vec_tile_shapes(1024, 2048) — same as relu, good for simple elementwise.
- Reshape: Standard 2D reshape pattern — correct.
- Kernels per call: 1
- Kernel name: tilefwk_0_mix_aic

## 4. Workarounds
None needed.

## 5. Known Limitations
FP16 only. 2-input elementwise; uses cache for repeated shapes (avoids re-allocating output tensor).

## 6. Performance (PyPTO 0.2.0 + torch 2.8.0 + torch_npu 2.8.0.post2 + CANN 9.0.0)
- B1 primary median: 53.418us
- B64 primary median: 63.308us
- Correctness: B1-B64 bitwise PASS
- Raw profiling: /tmp/opencode/prof_results/mul/
- Parsed profiling: /tmp/opencode/parsed_results/mul_parsed.json

## 7. Gate Status
- G8 (Correctness): PASS
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS

## 8. Review Notes
Good implementation with output tensor caching. _mul_cache pattern is good practice for repetitive calls.
