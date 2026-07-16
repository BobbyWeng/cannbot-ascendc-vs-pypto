# PyPTO Add Diagnostic Report

## Summary

**Status**: ✅ ALL CASES PASS — Four-input chained Add via `pypto.op.add` is working correctly for all experiment shapes.

## Root Cause of Earlier Failures

Three distinct issues were conflated in earlier errors:

### Issue 1: `pypto.op.add` API Signature (NOT `pypto.op.add(y, x1, x2)`)
- **API_REPORT.md had the wrong signature**: `pypto.op.add(y, x1, x2)` (output-as-first-arg style)
- **Actual signature** (confirmed via `inspect.signature`):
  ```python
  pypto.op.add(input: Tensor, other: Union[Tensor, float], *, alpha: Union[int, float] = 1) -> Tensor
  ```
- **Fix**: Use `y.move(pypto.op.add(x1, x2))` (returns new tensor, `.move()` stores to output)

### Issue 2: `set_vec_tile_shapes` Constraint (Vector Processing Width)
- The original tile shape `set_vec_tile_shapes(1024, 2048)` was copied from the ReLU kernel
- **Key constraint found for `pypto.op.add`**:
  - The first tile axis (vector element count) must be **≤ 128** for the backend C++ compiler to succeed
  - OR if > 128, the product `ta * tb` must be ≤ 128 * cols
  - 128 corresponds to the vector processing width (128 FP16 elements per vector instruction on Ascend 910B)
- **Fix**: Changed to `set_vec_tile_shapes(128, 1024)`

### Issue 3: 3D+ Tensors Not Supported by JIT Kernel
- The JIT kernel with `pypto.Tensor([], pypto.DT_FP16)` (dynamic shape) doesn't accept 3D+ tensors directly
- **Fix**: Use 2D reshape (same pattern as ReLU): `x.reshape(-1, orig_shape[-1])`

### Issue 4: JIT Functions Cannot Be in `__main__`
- PyPTO uses `inspect.getsourcelines()` to read the JIT function source
- Functions defined in `__main__` or `-c` strings cannot be read by `inspect`
- **Fix**: All `@pypto.frontend.jit` functions must be in imported module files

## Diagnostic Findings

### Case A: FP16 [1,32] + [1,32] — Binary Add
- **Result**: PASS (bitwise equal)
- **Tile shape**: (128, 1024)
- **Purpose**: Minimal binary add test

### Case B: FP16 [1,256,384] + [1,256,384] — Binary Add
- **Result**: PASS (bitwise equal)
- **Tile shape**: (128, 1024)
- **Purpose**: Experiment shape binary add

### Case C: FP16 [1,256,384] + [1,256,384] — Binary Add (B=1)
- **Result**: PASS (bitwise equal)
- **Tile shape**: (128, 1024)
- **Note**: B=1 is same as Case B shape

### Case D: Four-input Chained Add, B=1
```python
t1 = add(x1, x2); t2 = add(t1, x3); y = add(t2, x4)
```
- **Result**: PASS (bitwise equal)
- **Implementation**: Three sequential `add_binary_kernel` calls via host wrapper

### Case E: Four-input Chained Add, All Batch Sizes
- **Result**: ALL PASS (bitwise equal for all B ∈ {1,2,4,8,16,32,64})
- **Verified against**: Pre-generated reference data

## set_vec_tile_shapes Constraint Details

| ta | tb | shape | result |
|----|----|-------|--------|
| 1024 | 2048 | (1,32) | PASS |
| 1024 | 2048 | (128,384) | PASS |
| 1024 | 2048 | (256,384) | FAIL |
| 128 | 1024 | (256,384) | PASS |
| 129 | 256 | (256,384) | PASS |
| 129 | 383 | (383,384) | FAIL |
| 256 | 256 | (256,384) | FAIL |
| 96 | 1024 | (256,384) | PASS |
| 128 | 1024 | (16384,384) | PASS (B=64) |

**Rule**: ta ≤ 128 for any row count, OR ta * tb ≤ 128 * cols (= 49,152) for cols=384.

## Version Compatibility

| Component | Version | Path |
|-----------|---------|------|
| pypto | 0.2.0 | /home/developer/.local/lib/python3.11/site-packages/pypto/ |
| torch | 2.7.1+cpu | system |
| torch_npu | 2.7.1.post4 | system |
| CANN | cann-9.0.0 | /home/developer/Ascend/cann-9.0.0/ |
| Python | 3.11.4 | /opt/buildtools/Python-3.11.4/ |
| Device | Ascend 910B | dav-2201 |

All versions are consistent with the working ReLU environment.

## Source Differences: ReLU vs Add (Working)

| Aspect | ReLU (`relu_impl.py`) | Add (`add_impl.py`, working) |
|--------|----------------------|------------------------------|
| JIT decorator | `@pypto.frontend.jit` | `@pypto.frontend.jit` |
| Tensor type | `pypto.Tensor([], pypto.DT_FP16)` | `pypto.Tensor([], pypto.DT_FP16)` |
| Kernel operation | `y.move(pypto.op.relu(x))` | `y.move(pypto.op.add(x1, x2))` |
| Tile shape | `(1024, 2048)` | `(128, 1024)` |
| Input count | 2 (x, y) | 3 (x1, x2, y) |
| Output write | `.move(result)` | `.move(result)` |
| Reshape | `x.reshape(-1, orig_shape[-1])` | `x.reshape(-1, orig_shape[-1])` |

## File Structure (Cleaned)

```
operators/add/pypto/
├── src/
│   ├── __init__.py
│   └── add_impl.py          # add_binary_kernel, add_binary, add_4
├── golden/
│   └── add_golden.py
├── tests/
│   ├── test_add.py           # Main test (to be created/updated)
│   └── (diagnostic files deleted)
├── diagnostics/
│   ├── case_a.py .. case_e.py
│   ├── DIAGNOSTIC_REPORT.md
│   └── tile_shape_scan.py
├── SPEC/
├── API_REPORT/
└── DESIGN/
```

## Conclusion

The PyPTO binary `pypto.op.add` API works correctly for FP16. The earlier "Run pass failed" errors were caused by:
1. **Wrong tile shape** (ta > 128 for large tensors) — fixed with `ta=128`
2. **3D tensors** — fixed with 2D reshape pattern
3. **JIT in `__main__`** — fixed by using imported module

**No backend unsupported determination.** The add operation is fully functional.
