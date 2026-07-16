# Not (LogicalNot) Operator Specification

## Operator
- Name: Not (LogicalNot)
- Formula: Y = LogicalNot(X)
- Category: Element-wise logical

## Input/Output
| Name | Shape | dtype | Format |
|------|-------|-------|--------|
| x    | [B, 256, 384] | BOOL (uint8) | ND contiguous |
| y    | [B, 256, 384] | BOOL (uint8) | ND contiguous |

## Shape Constraints
- B ∈ {1, 2, 4, 8, 16, 32, 64}
- Total elements per batch: 256 * 384 = 98304
- No broadcasting
- BOOL stored as uint8: 0 = False, nonzero = True

## Precision Requirements
- Bitwise equality required
- Logical NOT: 0 -> 1, nonzero -> 0
