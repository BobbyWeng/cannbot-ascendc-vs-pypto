# API Exploration Report: Equal

## 1. Overview
Operator: Equal (element-wise comparison)
Formula: Y = Equal(X1, X2)
Target: Ascend 910B (dav-2201)

## 2. API Mapping
| Operation | PyPTO API | Status |
|-----------|-----------|--------|
| Element-wise compare | `pypto.eq(x1, x2) -> Tensor` | Available |

Confirmed: `pypto.eq` is available in `pypto.op.comparison` module and exported as `pypto.eq`.
Returns BOOL tensor with same shape as inputs.

## Constraints
- Input tensors must be contiguous ND format
- FP16 input, BOOL output supported
- Both inputs must have identical shape
- Tiling: `set_vec_tile_shapes(ta, tb)` with first axis `ta ≤ 128` for 2D tensors
- Kernel accepts 2D tensors only; 3D+ must be reshaped to 2D via host wrapper
- BOOL output uses `pypto.DT_BOOL`

## Example JIT Kernel
```python
@pypto.frontend.jit
def equal_kernel(x1: pypto.Tensor([], pypto.DT_FP16),
                 x2: pypto.Tensor([], pypto.DT_FP16),
                 y: pypto.Tensor([], pypto.DT_BOOL)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.eq(x1, x2))
```

## Feasibility
Fully feasible. pypto.eq supports FP16 element-wise comparison with BOOL output.

## Conclusion
Feasible with proper tiling configuration.
