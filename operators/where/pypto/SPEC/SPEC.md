# Where Operator Specification

## Operator
- Name: Where
- Formula: Y[i] = Condition[i] ? X1[i] : X2[i]
- Category: Element-wise select

## Input/Output
| Name | Shape | dtype | Format |
|------|-------|-------|--------|
| condition | [B, 256, 384] | BOOL (uint8) | ND contiguous |
| x1 | [B, 256, 384] | FP16 | ND contiguous |
| x2 | [B, 256, 384] | FP16 | ND contiguous |
| y  | [B, 256, 384] | FP16 | ND contiguous |

## Shape Constraints
- B ∈ {1, 2, 4, 8, 16, 32, 64}
- Total elements per batch: 256 * 384 = 98304
- All inputs must have identical shape (no broadcasting)
- Condition is uint8 (0 = false, non-zero = true)

## Precision Requirements
- FP16: bitwise equality with torch.where
- NaN/Inf in the non-selected branch must NOT propagate to output
- If Condition[i] is True, output[i] must equal X1[i] even if X2[i] is NaN
- If Condition[i] is False, output[i] must equal X2[i] even if X1[i] is NaN
