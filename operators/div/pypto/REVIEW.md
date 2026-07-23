# G13 Code Review: div (PyPTO)

**Date**: 2026-07-23T09:15:49.398831+00:00
**Reviewer**: pypto-op-orchestrator (automated review)
**Score**: 82/100 — PASS

## 1. Code Summary
Broadcast division kernel (x1/x2 with last-dim broadcast) using pypto.op.div with 2D reshape workaround.

## 2. API Correctness
pypto.op.div(x1, x2) correct for broadcast division when inputs are 2D and tile shapes are compatible.

## 3. Memory & Tile Configuration
- Tile: set_vec_tile_shapes(128, 1024) — must be <256 on first dim due to known PyPTO backend CompileFunction failure for tile[0]>=256.
- Reshape: Advanced 2D reshape handling broadcast dimensions. x2_shape[-1] may be 1 (broadcast).
- Kernels per call: 1
- Kernel name: tilefwk_0_mix_aic

## 4. Workarounds
Multiple: (1) 2D reshape to avoid 4D CompileFunction bug, (2) tile[0]=128<256 to avoid tile-size CompileFunction bug, (3) reciprocal+mul approach discarded due to FP16 precision loss.

## 5. Known Limitations
Backend restrictions force 2D+small-tile workaround. Broadcast only on last dim. FP16 only.

## 6. Performance (PyPTO 0.2.0 + torch 2.8.0 + torch_npu 2.8.0.post2 + CANN 9.0.0)
- B1 primary median: 55.128us
- B64 primary median: 129.387us
- Correctness: B1-B64 bitwise PASS
- Raw profiling: /tmp/opencode/prof_results/div/
- Parsed profiling: /tmp/opencode/parsed_results/div_parsed.json

## 7. Gate Status
- G8 (Correctness): PASS
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS

## 8. Review Notes
Most restricted operator. Three documented backend limitations required workarounds. Implementation is correct but fragile.
