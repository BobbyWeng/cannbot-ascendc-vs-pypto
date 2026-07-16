# API Exploration Report: Add (4-input)

## 1. Overview
Operator: Add (4-input element-wise addition)
Formula: Y = ((X1 + X2) + X3) + X4
Target: Ascend 910B (dav-2201)

## 3. API Mapping
| Operation | PyPTO API | Status |
|-----------|-----------|--------|
| Add | `pypto.op.add(x1, x2, *, alpha=1) -> Tensor` | Available |
| Add (chain) | `pypto.op.add(t, x3, *, alpha=1) -> Tensor` | Available |
| Add (chain) | `pypto.op.add(t, x4, *, alpha=1) -> Tensor` | Available |

## Constraints
- Input tensors must be contiguous ND format
- FP16 input/output supported
- All inputs must have identical shape
- Tiling: `set_vec_tile_shapes(ta, tb)` with first axis `ta ≤ 128` for 2D tensors
- Vector element count (`ta`) must be ≤ 128 to avoid backend pass failure
- Kernel accepts 2D tensors only; 3D+ must be reshaped to 2D via host wrapper

## Example JIT Kernel
```python
@pypto.frontend.jit
def add_binary_kernel(x1: pypto.Tensor([], pypto.DT_FP16),
                      x2: pypto.Tensor([], pypto.DT_FP16),
                      y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.op.add(x1, x2))
```

## Feasibility
Fully feasible. pypto.op.add supports FP16 element-wise addition.
Three chained adds with intermediate results kept in UB.

## Reference Implementations
- pypto.examples: chained element-wise add patterns available
- models/: not applicable for simple element-wise

## Risk Assessment
- UB pressure: 4 inputs + 1 temp + 1 output = 6 tiles per pipeline
- Must ensure vec_tile_shapes fit within UB capacity
- Double buffering may increase UB pressure

## Conclusion
Feasible with proper tiling configuration.
