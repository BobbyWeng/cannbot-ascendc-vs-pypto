# G13 Code Review: add (PyPTO)

**Date**: 2026-07-23T09:15:49.398831+00:00
**Reviewer**: pypto-op-orchestrator (automated review)
**Score**: 88/100 — PASS

## 1. Code Summary
4-input Add kernel using three chained pypto.op.add calls (x1+x2+x3+x4) with FP16 tensors.

## 2. API Correctness
pypto.op.add(x1, x2) correct. Chained pattern (add_binary called 3 times) handles 4-input case.

## 3. Memory & Tile Configuration
- Tile: set_vec_tile_shapes(128, 1024) — appropriate for multi-kernel pattern.
- Reshape: 2D reshape pattern with re-creation per call (no caching — simpler but slightly slower).
- Kernels per call: 3
- Kernel name: tilefwk_0_mix_aic

## 4. Workarounds
None needed.

## 5. Known Limitations
4 inputs → 3 binary ops → 3× kernel launch overhead. Intermediate tensors not cached (re-created each call).

## 6. Performance (PyPTO 0.2.0 + torch 2.8.0 + torch_npu 2.8.0.post2 + CANN 9.0.0)
- B1 primary median: 138.23649999999998us
- B64 primary median: 150.275us
- Correctness: B1-B64 bitwise PASS
- Raw profiling: /tmp/opencode/prof_results/add/
- Parsed profiling: /tmp/opencode/parsed_results/add_parsed.json

## 7. Gate Status
- G8 (Correctness): PASS
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS

## 8. Review Notes
Multi-kernel approach is correct but adds overhead. Could potentially fuse if PyPTO supports multi-output.
