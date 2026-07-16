# API Exploration Report: Where

## 1. Overview
Operator: Where (element-wise ternary select)
Formula: Y[i] = Condition[i] ? X1[i] : X2[i]
Target: Ascend 910B (dav-2201)

## 2. API Mapping
| Operation | PyPTO API | Status |
|-----------|-----------|--------|
| Where | `pypto.op.where(condition, x1, x2) -> Tensor` | Available |
| Where (cond bool) | `pypto.op.where(cond, x1, x2, *, round_mode=CAST_NONE)` | Available |

## Constraints
- Input tensors must be contiguous ND format
- FP16 input/output supported; condition as uint8/BOOL
- All inputs must have identical shape; no broadcasting
- Tiling: `set_vec_tile_shapes(ta, tb)` with first axis `ta ≤ 128` for 2D tensors
- Vector element count (`ta`) must be ≤ 128 to avoid backend pass failure
- Kernel accepts 2D tensors only; 3D+ must be reshaped to 2D via host wrapper

## Example JIT Kernel
```python
@pypto.frontend.jit
def where_kernel(condition: pypto.Tensor([], pypto.DT_UINT8),
                 x1: pypto.Tensor([], pypto.DT_FP16),
                 x2: pypto.Tensor([], pypto.DT_FP16),
                 y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.op.where(condition, x1, x2))
```

## Feasibility
Fully feasible. pypto.op.where supports FP16 element-wise conditional select with uint8 condition.

## Risk Assessment
- Condition type is uint8 — verify PyPTO handles non-bool condition correctly
- NaN propagation: PyPTO where must only select from active branch
- UB pressure: 3 inputs (cond+2) + 1 output = 4 tiles per pipeline

## Conclusion
Feasible with proper tiling configuration.
