# Add (4-input) Operator Design

## API Selection
- Primary: pypto.op.add — element-wise addition
- Three chained calls for ((X1+X2)+X3)+X4

## Shape
- Input/output: [B, 256, 384] FP16 contiguous ND
- Total elements = B * 256 * 384 = B * 98304

## Tiling Strategy
- Vector operation: set_vec_tile_shapes for pure element-wise
- Tile shape determined by UB capacity (4 inputs + 1 temp + 1 output)
- Multi-core parallelization handled by PyPTO runtime

## Loop Structure
```
for each tile:
    tmp = add(x1_tile, x2_tile)
    tmp = add(tmp, x3_tile)
    y_tile = add(tmp, x4_tile)
```

## Special Handling
- Intermediate results stay in UB (no GM round-trip)
- Chain order strictly follows ((X1+X2)+X3)+X4

## UB Budget
- 4 input tiles + 1 tmp tile + 1 output tile
- Double buffering: 6 * tile_size * 2 bytes * 2 = 24 * tile_size bytes
- With 8192 elements/tile: 24 * 8192 * 2 = 393 KB
- UB on 910B: ~512 KB → feasible with TILE_LENGTH up to 8192
