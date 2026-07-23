# G13 Code Review: relu (PyPTO)

**Date**: 2026-07-23T09:15:49.398831+00:00
**Reviewer**: pypto-op-orchestrator (automated review)
**Score**: 95/100 — PASS

## 1. Code Summary
ReLU activation kernel using pypto.op.relu on FP16 tensors.

## 2. API Correctness
pypto.op.relu(x) correctly implements ReLU. FP16 input/output.

## 3. Memory & Tile Configuration
- Tile: set_vec_tile_shapes(1024, 2048) — largest tile, efficient for simple single-input ops.
- Reshape: Standard 2D reshape pattern — correct.
- Kernels per call: 1
- Kernel name: tilefwk_0_mix_aic

## 4. Workarounds
None. op.relu works correctly under torch 2.8.0+torch_npu 2.8.0 (note: was SIGSEGV under torch 2.7.1).

## 5. Known Limitations
FP16 only. For FP32 ReLU, separate kernel needed.

## 6. Performance (PyPTO 0.2.0 + torch 2.8.0 + torch_npu 2.8.0.post2 + CANN 9.0.0)
- B1 primary median: 48.759us
- B64 primary median: 56.448us
- Correctness: B1-B64 bitwise PASS
- Raw profiling: /tmp/opencode/prof_results/relu/
- Parsed profiling: /tmp/opencode/parsed_results/relu_parsed.json

## 7. Gate Status
- G8 (Correctness): PASS
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS

## 8. Review Notes
Cleanest implementation. Large tile shape (1024,2048) optimal for this simple elementwise op.
