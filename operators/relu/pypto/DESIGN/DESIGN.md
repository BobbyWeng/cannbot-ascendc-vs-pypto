# ReLU Operator Design

## API Selection
- Primary: pypto.op.relu(x) — built-in element-wise activation
- Alternative: pypto.maximum(x, 0) — functionally equivalent

## Shape
- Input/output: [B, 12, 256, 32] FP16 contiguous ND
- Total elements = B * 12 * 256 * 32 = B * 98304

## Tiling Strategy
- Use default vec tile shapes via pypto.set_vec_tile_shapes
- Multi-core parallelization handled by PyPTO runtime
- Batch dimension maps to outermost loop

## Loop Structure
```
for batch in range(B):
    output_tile = pypto.op.relu(input_tile)
```

## Special Handling
- No edge cases for FP16 (no NaN propagation concerns)
- Negative zero treated as zero (max(0, -0.0) = 0.0)
