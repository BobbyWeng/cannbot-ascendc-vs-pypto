# G13 Code Review: where (PyPTO)

**Date**: 2026-07-23T09:15:49.398831+00:00
**Reviewer**: pypto-op-orchestrator (automated review)
**Score**: 85/100 — PASS

## 1. Code Summary
Conditional select kernel (torch.where equivalent) using pypto.where with bool condition and FP16 data paths.

## 2. API Correctness
pypto.where(condition, x1, x2) with DT_BOOL condition and DT_FP16 data — correct type mapping.

## 3. Memory & Tile Configuration
- Tile: set_vec_tile_shapes(64, 256) — smaller tile than other ops, accommodates 3-input complexity.
- Reshape: Standard 2D reshape with bool conversion workaround (uint8 → bool).
- Kernels per call: 1
- Kernel name: tilefwk_0_mix_aic

## 4. Workarounds
PyPTO backend TiledWhereOperation has uint8 expansion bug. Workaround: convert uint8 to bool with .bool() before kernel call. This is documented in source code comments.

## 5. Known Limitations
Condition tensor consumes memory bandwidth but not compute. Larger B sees significant scaling (2.8x).

## 6. Performance (PyPTO 0.2.0 + torch 2.8.0 + torch_npu 2.8.0.post2 + CANN 9.0.0)
- B1 primary median: 75.96799999999999us
- B64 primary median: 215.385us
- Correctness: B1-B64 bitwise PASS
- Raw profiling: /tmp/opencode/prof_results/where/
- Parsed profiling: /tmp/opencode/parsed_results/where_parsed.json

## 7. Gate Status
- G8 (Correctness): PASS
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS

## 8. Review Notes
Highest latency among elementwise ops due to conditional execution path. Bool conversion workaround is stable.
