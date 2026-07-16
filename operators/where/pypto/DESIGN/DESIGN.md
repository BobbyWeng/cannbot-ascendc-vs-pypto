# Where Operator Design

## API Selection
- Primary: pypto.op.where — element-wise conditional select
- Signature: where(condition, x1, x2) → y

## Shape
- Input: condition [B, 256, 384] uint8, x1/x2 [B, 256, 384] FP16
- Output: [B, 256, 384] FP16 contiguous ND
- Total elements = B * 256 * 384 = B * 98304

## Tiling Strategy
- Vector operation: set_vec_tile_shapes for pure element-wise select
- 3 inputs (cond + x1 + x2) + 1 output = 4 tiles per pipeline stage
- Multi-core parallelization handled by PyPTO runtime

## Loop Structure
```
for each tile:
    y_tile = where(cond_tile, x1_tile, x2_tile)
```

## Special Handling
- Condition is uint8 (BOOL); PyPTO where must handle non-bool condition
- NaN/Inf in non-selected branch must NOT propagate
- 3D+ tensors reshaped to 2D for kernel, then reshaped back

## UB Budget
- 3 input tiles (cond uint8 small) + 1 output tile
- Double buffering: 4 * tile_size * 2 bytes * 2 = 16 * tile_size bytes
- With 8192 elements/tile: 16 * 8192 * 2 = 262 KB (plus cond at 8KB)
- Total ~270 KB, fits in UB on 910B (~512 KB)
