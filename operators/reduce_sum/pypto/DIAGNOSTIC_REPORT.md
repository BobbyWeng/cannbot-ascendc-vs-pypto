# ReduceSum PyPTO Diagnostic Report

## Status: FIXED

### Issue History
1. **`DT_FLOAT16` not found** — PyPTO uses `DT_FP16`, not `DT_FLOAT16`
2. **`pypto.op.reduce_sum` not found** — PyPTO uses `pypto.op.sum(input, dim, keepdim)`, not `reduce_sum`

### Current Implementation
- JIT kernel: `reduce_sum_kernel(x: FP16, y: FP16)` with `pypto.op.sum(x, dim=-1)`
- All shapes work correctly: [1,1,32], [1,1,128], [1,256,384]
- Precision: FP16 accumulation, max_diff ~0.06 vs FP32 golden (within FP16 expectation)

### Verification
```
[1,1,32] -> [1,1]: max_diff=0.0 (bitwise)
[1,1,128] -> [1,1]: max_diff=0.0137
[1,256,384] -> [1,256]: max_diff=0.0625
```
