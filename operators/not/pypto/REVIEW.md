# G13 Code Review: not (PyPTO)

**Date**: 2026-07-23T09:15:49.398831+00:00
**Reviewer**: pypto-op-orchestrator (automated review)
**Score**: 92/100 — PASS

## 1. Code Summary
Logical NOT kernel for BOOL (uint8) tensors using pypto.op.logical_not.

## 2. API Correctness
pypto.op.logical_not(x) is correct. Input/output dtype DT_BOOL (uint8).

## 3. Memory & Tile Configuration
- Tile: set_vec_tile_shapes(128, 1024) — appropriate for vector elementwise.
- Reshape: Standard 2D reshape (-1, orig_shape[-1]) pattern — correct for elementwise ops.
- Kernels per call: 1
- Kernel name: tilefwk_0_mix_aic

## 4. Workarounds
None needed. pypto.Tensor([], DT_BOOL) annotation works correctly under PyPTO 0.2.0.

## 5. Known Limitations
Only supports uint8/bool input. Other dtypes need separate kernel.

## 6. Performance (PyPTO 0.2.0 + torch 2.8.0 + torch_npu 2.8.0.post2 + CANN 9.0.0)
- B1 primary median: 57.048us
- B64 primary median: 91.178us
- Correctness: B1-B64 bitwise PASS
- Raw profiling: /tmp/opencode/prof_results/not/
- Parsed profiling: /tmp/opencode/parsed_results/not_parsed.json

## 7. Gate Status
- G8 (Correctness): PASS
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS

## 8. Review Notes
Clean implementation. Flat scaling (1.6x from B1 to B64) indicates good tile utilization.
