# Mul Design

## API Choice
- Primary: `pypto.op.mul(x1, x2)` via `@pypto.frontend.jit`

## Tiling Strategy
- Default vec tile shapes (1024, 2048)
- PyPTO runtime handles multi-core parallelization
- 2D reshape for JIT: [-1, 32] (last dim = 32)

## Loop Structure
- Batch loop: handled by PyPTO runtime
- Per-tile: `pypto.op.mul(input_tile_x1, input_tile_x2)`

## Special Cases
- FP16 overflow: values > 65504 saturate to Inf
- Signed zero: IEEE 754 requires +0 × -0 = -0
- No NaN quantization issues
