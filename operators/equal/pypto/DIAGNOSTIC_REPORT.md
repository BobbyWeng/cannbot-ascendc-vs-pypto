# PyPTO Equal Blocker Diagnostic Report

## Status: COMPLETE (was BLOCKED_BACKEND_EQUAL)

## Root Cause

The BLOCKED_BACKEND_EQUAL status had **two independent issues**:

### Issue 1: Wrong output dtype (FP16 instead of BOOL)

`pypto.eq()` returns a **packed bitmask** when the output tensor is declared as `DT_FP16`. The result values are subnormal FP16 values (e.g. `0x0101`, `0x0100`, `0x0001`) which look like packed comparison results, not proper 0.0/1.0 booleans. When the output tensor is declared as `DT_BOOL`, `pypto.eq()` produces correct boolean results.

**Fix**: Changed the kernel signature from `y: pypto.Tensor([], pypto.DT_FP16)` to `y: pypto.Tensor([], pypto.DT_BOOL)`, and the wrapper's output tensor from `dtype=torch.float16` to `dtype=torch.bool`.

### Issue 2: CompileFunction error with ta > 64

When `set_vec_tile_shapes(ta, tb)` uses `ta > 64`, the backend's CompileFunction pass fails with `Errcode: FFFFFF!`. This is an independent backend limitation. The threshold is exactly 64 (works: 8, 16, 32, 64; fails: 96, 128).

**Fix**: Changed `set_vec_tile_shapes(128, 1024)` to `set_vec_tile_shapes(64, 1024)`.

## Test Results (After Fix)

All 7 batch sizes (B=1,2,4,8,16,32,64) pass with bitwise equality:
- B=1: PASS (114 equal elements)
- B=2: PASS (136 equal elements)
- B=4: PASS (164 equal elements)
- B=8: PASS (224 equal elements)
- B=16: PASS (338 equal elements)
- B=32: PASS (688 equal elements)
- B=64: PASS (1265 equal elements)

## Changes Made

| File | Change |
|------|--------|
| `pypto/src/equal_impl.py` | `DT_FP16` → `DT_BOOL` for kernel output, `torch.float16` → `torch.bool` for wrapper output, `set_vec_tile_shapes(128, 1024)` → `set_vec_tile_shapes(64, 1024)` |

## Remaining Notes

- The `ta > 64` CompileFunction limitation is a backend issue that affects BOOL output kernels.
- Previous incorrect conclusion that "this is not fixable from user code" was wrong — the fix was to correctly declare the output dtype as BOOL and use a compatible tile shape.
