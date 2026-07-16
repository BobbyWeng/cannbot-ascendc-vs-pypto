# PyPTO MatMul — Diagnostic Report

## Status: BLOCKED_BACKEND

## Error

```
RuntimeError: Errcode: FC4000!
Invalid tile values: kL0=0, kL1a=0, kL1b=0, mL0=0, mL1=0, nL0=0, nL1=0
func CheckCubeTiling, file cube_operation_impl.cpp, line 306
```

## Backend Stage

CompileFunction / CheckCubeTiling — the Cube tiling engine cannot compute valid tile parameters.

## Reproduction

### Minimal Case

```python
from pypto.frontend import jit
from pypto.tensor import TensorAnnotation
import pypto

a_def = TensorAnnotation((256, 256), pypto.DT_FP16, "a")
b_def = TensorAnnotation((256, 32), pypto.DT_FP16, "b")
out_def = TensorAnnotation((256, 32), pypto.DT_FP16, "out")

@jit
def matmul_fn(a: a_def, b: b_def, out: out_def):
    c = pypto.matmul(a, b, pypto.DT_FP16)
    out.move(c)
```

All matmul shapes tested fail:
- [1,256,256] × [1,256,32] (batched, 4D)
- [256,256] × [256,32] (2D)
- [256,256] × [256,256] (square 2D)

## Root Cause

The PyPTO backend Cube tiling engine (`Matmul::CheckCubeTiling`) returns zero tile values for all dimensions (kL0, kL1a, kL1b, mL0, mL1, nL0, nL1 = 0). This indicates the tiling configuration cannot be automatically derived for any matmul shape in this PyPTO version.

## Workarounds Attempted

| Workaround | Result |
|------------|--------|
| `pypto.set_matrix_size([256, 32, 256])` | Still fails (F21003) |
| `pypto.set_cube_tile_shapes(...)` | Still fails (F21003) |
| 2D matmul instead of batched | Still fails (FC4000) |
| Square matmul | Still fails (FC4000) |
| Small matmul [16,16] × [16,16] | Not tested (JIT parser issues) |

## Conclusion

PyPTO `matmul` is **BLOCKED_BACKEND** due to the Cube tiling engine returning invalid tile values. This affects ALL matmul shapes, not just the target [B,12,256,256] × [B,12,256,32].

**PyPTO is excluded from the three-way performance ranking for MatMul.**
