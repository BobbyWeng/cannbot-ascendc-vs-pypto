# API Exploration Report — Div (Broadcast)

## API Mapping

| Operation | PyPTO API | Status | Notes |
|-----------|-----------|--------|-------|
| Element-wise divide | `pypto.op.div(x1, x2)` | Primary path | Must verify broadcast support |
| Element-wise divide | `pypto.op.div(x1, x2)` via `@pypto.frontend.jit` | JIT-compiled | Uses 2D tensor view |
| Reciprocal | `pypto.op.reciprocal(x)` | Fallback path | Available as built-in |
| Multiply | `pypto.op.mul(x1, x2)` | Fallback path | Available as built-in |
| Reciprocal + Mul decomposition | `pypto.op.mul(x1, pypto.op.reciprocal(x2))` | Fallback strategy | If native Div not supported |

## Feasibility
- **Div**: Supported as built-in API
- **Reciprocal**: Supported as built-in API
- **Broadcast**: PyPTO supports element-wise broadcast for 4D tensors when dimensions align
- **Dtype**: FP16 supported for both Div and Reciprocal
- **Vectorization**: Automatic via PyPTO JIT

## Constraints
- Inputs must be contiguous ND tensors
- dtype must match (both FP16)
- Broadcast: X2 last dim = 1 broadcasts to match X1 last dim = 256
- PyPTO tensor rank limited to 4 (kernel shapes are 4D)
- set_vec_tile_shapes must be called before vector operations

## Implementation Strategy Decision

### Strategy A: Native Div (Preferred)
```python
y = pypto.op.div(x1, x2)
```
- Status: Primary path
- Requires: Working broadcast in PyPTO runtime

### Strategy B: Reciprocal + Mul (Fallback)
```python
y = pypto.op.mul(x1, pypto.op.reciprocal(x2))
```
- Status: Fallback if native Div fails
- Note: Must be documented as "reciprocal_mul", NOT native Div

### Strategy C: Materialized Broadcast (Last Resort)
- Expand X2 to full [B,12,256,256] before Div
- NOT for primary comparison ranking
- Marked as materialized-broadcast fallback

## Evidence
- pypto.op.div documented in operation index
- pypto.op.reciprocal documented and tested in Mul experiment
- pypto.op.mul confirmed working (from Mul experiment)
- Broadcast behavior TBD from profiling
