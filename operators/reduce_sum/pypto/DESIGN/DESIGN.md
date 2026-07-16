# ReduceSum Operator Design

## API Selection
- Primary: pypto.op.reduce_sum — reduction sum over last dimension
- Fallback: If unavailable, document as BLOCKED_BACKEND

## Shape
- Input: [B, 256, 384] FP16 contiguous ND
- Output: [B, 256] FP16 contiguous ND
- Reduction axis = -1 (dim 2), keepdims = false

## Tiling Strategy
- Vector operation: set_vec_tile_shapes for reduction
- Tile shape determined by UB capacity
- Multi-core parallelization handled by PyPTO runtime

## Loop Structure
```
for each input tile:
    y = reduce_sum(x, axis=-1)
```

## Accumulation Dtype
- Preferred: FP32 accumulation for better accuracy
- Cast to FP16 for output

## UB Budget
- 1 input tile + 1 output tile
- Double buffering
- FP16: 2 bytes/element

## Risk Assessment
- Reduction API may not be available in PyPTO
- If BLOCKED_BACKEND, produce diagnostic report
