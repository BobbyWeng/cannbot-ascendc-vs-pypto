# Not (LogicalNot) Operator Design

## API Selection
- Primary: pypto.op.logical_not — element-wise logical NOT

## Shape
- Input/output: [B, 256, 384] BOOL (uint8) contiguous ND
- Total elements = B * 256 * 384 = B * 98304

## Tiling Strategy
- Vector operation: set_vec_tile_shapes for pure element-wise
- Tile shape determined by UB capacity (1 input + 1 output)
- Multi-core parallelization handled by PyPTO runtime

## Loop Structure
```
for each tile:
    y_tile = logical_not(x_tile)
```

## Special Handling
- BOOL stored as uint8: 0 = False, nonzero = True
- logical_not maps: 0 -> 1, nonzero -> 0

## UB Budget
- 1 input tile + 1 output tile
- Double buffering: 2 * tile_size * 1 byte * 2 = 4 * tile_size bytes
- With 8192 elements/tile: 4 * 8192 = 32 KB
- UB on 910B: ~512 KB → easily feasible
