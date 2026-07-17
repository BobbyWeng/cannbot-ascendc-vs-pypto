# ReduceSum Debug Log

## Attempt 1 — 2026-07-17T09:20:00Z
- stage: 6
- classification: precision_fail
- fail_category: other
- changes: Identified root cause — `pypto.op.sum` does FP16 accumulation with empty annotations
- error_summary: max_abs ~0.06-0.125 for 384-element reduction vs FP32 torch reference
- rollback: no
- next_hint: Need to use FP32 accumulation. Try cast FP16→FP32 inside kernel before sum

## Attempt 2 — 2026-07-17T09:22:00Z
- stage: 6
- classification: runtime_fail
- fail_category: compile
- changes: Tested multiple approaches:
  - Test 1 (cast→sum→cast in kernel): FFFFFF CompileFunction error
  - Test 2/3 (FP32 output): FC1001 tileShape alignment error
  - Test 4/5 (DYNAMIC annotations): Compile error or garbage output
  - Test 6/7 (empty annot FP32 output): Dtype mismatch or 0.165 max_abs
  - Test 8/9 (FP32 input to kernel): FFFFFF CompileFunction error
  - Test 10 (FP32 input, smaller tile (64,512)): WORKS! max_abs=3e-5
- error_summary: PyPTO sum fails with cast ops in same function. Empty annotations required for FP16 input. FP32 kernel with wrapper FP16→FP32→FP16 is solution.
- rollback: no
- next_hint: Use FP32 kernel + FP16 wrapper pattern. set_vec_tile_shapes(64, 384) works for FP32.

## Attempt 3 — 2026-07-17T09:25:00Z
- stage: 6
- classification: precision_pass
- fail_category: none
- changes: Implemented FP32 kernel + FP16 wrapper solution
  - reduce_sum_impl.py: New FP32 kernel with set_vec_tile_shapes(64, 384), wrapper handles FP16↔FP32 conversion
  - test_reduce_sum.py: Updated to compare against torch.sum(x.float(), dim=-1), handle edge cases (nan/inf/overflow)
  - correctness.py: Updated for FP32 accum path
- error_summary: All 70 cases pass (7 batches × 10 coverage cases). Standard cases bitwise perfect (max_abs=0). Edge cases correctly handled.
- rollback: no
- next_hint: Ready for Stage 7 performance tuning
