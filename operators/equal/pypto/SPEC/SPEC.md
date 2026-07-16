# Equal Operator Specification

## Operator
- Name: Equal
- Formula: Y = Equal(X1, X2)
- Category: Element-wise comparison

## Input/Output
| Name | Shape | dtype | Format |
|------|-------|-------|--------|
| x1   | [B, 256, 384] | FP16 | ND contiguous |
| x2   | [B, 256, 384] | FP16 | ND contiguous |
| y    | [B, 256, 384] | BOOL | ND contiguous |

## Shape Constraints
- B ∈ {1, 2, 4, 8, 16, 32, 64}
- Total elements per batch: 256 * 384 = 98304
- Both inputs must have identical shape and dtype
- No broadcasting

## Precision Requirements
- BOOL output: bitwise equality with torch.eq
- NaN semantics: NaN != NaN (consistent with torch.eq)
- +0.0 == -0.0 (per IEEE 754)
- Special values: Inf, -Inf, NaN, +/-0 must be handled correctly
