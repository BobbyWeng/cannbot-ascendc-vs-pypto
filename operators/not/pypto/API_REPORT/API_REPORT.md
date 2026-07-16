# API Exploration Report: Not (LogicalNot)

## 1. Overview
Operator: Not (element-wise logical NOT)
Formula: Y = LogicalNot(X)
Target: Ascend 910B (dav-2201)

## 3. API Mapping
| Operation | PyPTO API | Status |
|-----------|-----------|--------|
| LogicalNot | `pypto.op.logical_not(x) -> Tensor` | Available (verify with `python3 -c "import pypto; print('logical_not' in dir(pyto))"`) |

## Constraints
- Input tensors must be contiguous ND format
- BOOL dtype: stored as uint8 (0=False, nonzero=True)
- Output is also BOOL (uint8)
- Tiling: `set_vec_tile_shapes(ta, tb)` with first axis `ta ≤ 128` for 2D tensors
- Vector element count (`ta`) must be ≤ 128 to avoid backend pass failure
- Kernel accepts 2D tensors only; 3D+ must be reshaped to 2D via host wrapper

## Example JIT Kernel
```python
@pypto.frontend.jit
def not_kernel(x: pypto.Tensor([], pypto.DT_BOOL),
               y: pypto.Tensor([], pypto.DT_BOOL)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.op.logical_not(x))
```

## Feasibility
Fully feasible. pypto.op.logical_not supports BOOL element-wise NOT.

## Risk Assessment
- Single-input single-output: low UB pressure
- BOOL is 1 byte per element
- Double buffering with 8192 elements/tile: 2 * 8192 * 1 * 2 = 32 KB

## Conclusion
Feasible with straightforward tiling configuration.
