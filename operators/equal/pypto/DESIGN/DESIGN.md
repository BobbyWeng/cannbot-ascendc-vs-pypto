# Equal Operator Design

## API Selection
- Primary: pypto.eq — element-wise comparison returning BOOL tensor

## Shape
- Input: [B, 256, 384] FP16 contiguous ND, no broadcast
- Output: [B, 256, 384] BOOL contiguous ND
- Total elements = B * 256 * 384 = B * 98304

## Tiling Strategy
- Vector operation: set_vec_tile_shapes for pure element-wise comparison
- BOOL output is 1 byte per element vs 2 bytes for FP16 input
- Multi-core parallelization handled by PyPTO runtime

## Loop Structure
```
for each tile:
    y_tile = eq(x1_tile, x2_tile)
```

## Special Handling
- NaN != NaN must be preserved (torch.eq semantics)
- +0.0 == -0.0 per IEEE 754
- Intermediate results stay in UB (no GM round-trip)

## UB Budget
- 2 input tiles (FP16, 2 bytes each) + 1 output tile (BOOL, 1 byte)
- With TILE_LENGTH up to 8192: 3 * 8192 * 2 = 49152 bytes (approx 48 KB)
- Well within UB capacity on 910B (~512 KB)
