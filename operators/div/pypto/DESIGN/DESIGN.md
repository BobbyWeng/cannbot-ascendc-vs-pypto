# Div Design — PyPTO Broadcast Implementation

## API Choice
- **Primary**: `pypto.op.div(x1, x2)` via `@pypto.frontend.jit`
- **Fallback**: `pypto.op.mul(x1, pypto.op.reciprocal(x2))`
- **Last resort**: Materialized broadcast (expand X2 before div)

## Data Specification

### Kernel Signature
```
x1: [B, 12, 256, 256], FP16
x2: [B, 12, 256, 1],   FP16
y:  [B, 12, 256, 256], FP16
```

### 2D JIT Reshape
Since PyPTO JIT uses 2D tensors with tile shapes:
```
x1_2d: [-1, 256], FP16
x2_2d: [-1, 1],   FP16
y_2d:  [-1, 256], FP16
```

## Tiling Strategy
- Default vec tile shapes (1024, 2048)
- PyPTO runtime handles multi-core parallelization
- 2D reshape for JIT: [-1, 256] (last dim = 256)

## Broadcast Handling
- X2 has last dim = 1, which should broadcast to 256
- PyPTO runtime responsible for broadcast lowering
- If broadcast not supported, X2 must be manually repeated

## Loop Structure
- Batch loop: handled by PyPTO runtime
- Per-tile: `pypto.op.div(input_tile_x1, input_tile_x2)` with broadcast

## Special Cases
- Division by zero: results vary by implementation
- NaN/Inf propagation per IEEE 754
- Signed zero: IEEE 754 requires proper handling
- FP16 overflow: values > 65504 saturate to Inf
- Subnormal flushing: implementation-dependent
