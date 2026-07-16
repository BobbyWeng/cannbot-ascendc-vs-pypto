# API Exploration Report — Mul

## API Mapping
| Operation | PyPTO API | Status |
|-----------|-----------|--------|
| Element-wise multiply | `pypto.op.mul(x1, x2)` | Primary |
| Element-wise multiply | `pypto.op.mul(x1, x2)` via `@pypto.frontend.jit` | JIT-compiled |

## Feasibility
- **Supported**: Yes, element-wise Mul is a built-in operation in PyPTO
- **Dtype**: FP16 supported
- **Vectorization**: Automatic via PyPTO JIT

## Constraints
- Inputs must be contiguous ND tensors
- dtype must match (both FP16 in this case)
- Both inputs must have the same shape (no broadcasting in this experiment)
