# Design: Or (LogicalOr)

## API Mapping
| Operation | PyPTO API |
|-----------|-----------|
| Logical OR | `pypto.op.logical_or(x1, x2)` |

## Tile Sizes
| Parameter | Value |
|-----------|-------|
| vec_tile_shapes | (128, 1024) |

## Data Flow
1. Load X1 (bool/uint8) from GM to Local
2. Load X2 (bool/uint8) from GM to Local
3. Compute Y = logical_or(X1, X2)
4. Store Y from Local to GM

## Loop Structure
Single tile loop over the flattened 2D tensor.
- Reshape [B, 256, 384] to [-1, 384]
- Process row by row with tile size

## Risk Points
| Risk | Mitigation |
|------|------------|
| BOOL stored as uint8 | Use uint8 dtype throughout |
| Non-zero input values | logical_or handles any non-zero as True |
