# API Exploration Report: ReduceSum

## 1. Overview
Operator: ReduceSum (sum reduction over last dimension)
Formula: Y[b,i] = sum_j X[b,i,j]
Target: Ascend 910B (dav-2201)

## 2. API Mapping
| Operation | PyPTO API | Status |
|-----------|-----------|--------|
| ReduceSum on last dim | `pypto.op.reduce_sum(x, axis=-1) -> Tensor` | UNVERIFIED — may not exist as direct API |

## 3. Constraints
- Input tensors must be contiguous ND format
- FP16 input with FP32 accumulation preferred for accuracy
- axis=-1, keepdims=false
- Tiling: set_vec_tile_shapes(128, 1024) for 2D tensors
- Kernel accepts 2D tensors only; 3D+ must be reshaped to 2D

## 4. Backend Limitation
If pypto.op.reduce_sum is NOT available:
- Mark as BLOCKED_BACKEND
- Produce DIAGNOSTIC_REPORT.md with details
- PyPTO does not enter three-way performance ranking

## 5. Alternative
If reduce_sum not available, possible workaround:
- Use pypto.op.mul with mask, or element-wise ops
- But reduction semantics require backend support

## 6. Feasibility
Pending verification of pypto.op.reduce_sum availability.
