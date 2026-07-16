# Add Operator Specification (4-input)

## Operator
- Name: Add
- Formula: Y = ((X1 + X2) + X3) + X4
- Category: Element-wise arithmetic

## Input/Output
| Name | Shape | dtype | Format |
|------|-------|-------|--------|
| x1 | [B, 256, 384] | FP16 | ND contiguous |
| x2 | [B, 256, 384] | FP16 | ND contiguous |
| x3 | [B, 256, 384] | FP16 | ND contiguous |
| x4 | [B, 256, 384] | FP16 | ND contiguous |
| y  | [B, 256, 384] | FP16 | ND contiguous |

## Shape Constraints
- B ∈ {1, 2, 4, 8, 16, 32, 64}
- Total elements per batch: 256 * 384 = 98304
- All inputs must have identical shape and dtype
- No broadcasting

## Precision Requirements
- FP16: bitwise equality with fixed left-associative order
- Computation order: t1 = (X1 + X2), t2 = (t1 + X3), Y = (t2 + X4)
