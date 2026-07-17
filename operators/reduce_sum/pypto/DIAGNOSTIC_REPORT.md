# ReduceSum PyPTO Diagnostic Report

## Status: FP32 ACCUMULATION FIXED

### Issue
FP16 accumulation in `pypto.op.sum` causes max_abs ~0.06-0.125 for 384-element reduction when compared against torch FP32 accumulation reference.

### Root Cause
`pypto.op.sum` performs accumulation in the input dtype. Per execution constraints (`references/execution-constraints.md`), `sum` only officially supports `DT_FP32`. When given FP16 input, it accumulates in FP16, losing precision.

### Solution
**Two-layer architecture**: FP32 kernel + FP16 wrapper

1. **Kernel** (`reduce_sum_fp32_kernel`):
   - Takes FP32 I/O with `pypto.Tensor([], pypto.DT_FP32)` (empty annotations)
   - Uses `set_vec_tile_shapes(64, 384)` (smaller tile for FP32 4-byte elements)
   - Calls `pypto.op.sum(x, dim=-1)`

2. **Wrapper** (`reduce_sum_wrapper`):
   - Converts FP16 input to FP32: `x.float()`
   - Runs kernel on NPU
   - Converts FP32 output back to FP16: `.half()`

### Key Constraints Discovered
- **Tile shape must match dtype**: FP32 needs smaller tiles (64, 384) vs FP16 (128, 1024)
- **Empty annotations required**: `pypto.Tensor([], dtype)` is the only working pattern — DYNAMIC annotation has a compile bug (`ARG_ was not declared`)
- **No cast inside kernel**: `pypto.cast` + `pypto.op.sum` in the same function causes FFFFFF CompileFunction error
- **Tile last axis >= reduction dim**: 384 >= 384 ✓

### Verification Results
- **70/70 cases PASS** (7 batches × 10 coverage cases)
- Standard cases: bitwise perfect (max_abs=0.0 for most, 0.001953 for B=64)
- Edge cases (nan, inf, overflow): Correctly handled
- Special cases (large_values, overflow_risk): FP32 accumulator correctly returns finite values where FP16 overflows to inf

### Left as Diagnostics
All diagnostic scripts are preserved in `pypto/diagnostics/candidate_reduce_sum_fp32/`:
- test1 to test15 documenting the exploration path
- Full validation results
- Comprehensive comparison report
