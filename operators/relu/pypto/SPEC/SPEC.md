# ReLU Operator Specification

## Operator
- Name: ReLU
- Formula: y = max(x, 0)
- Category: Element-wise activation

## Input/Output
| Name | Shape | dtype | Format |
|------|-------|-------|--------|
| x | [B, 12, 256, 32] | FP16 | ND contiguous |
| y | [B, 12, 256, 32] | FP16 | ND contiguous |

## Shape Constraints
- B ∈ {1, 2, 4, 8, 16, 32, 64}
- Total elements per batch: 12 * 256 * 32 = 98304
- Input and output must have identical shape and dtype

## Precision Requirements
- FP16: bitwise equality (atol=0, rtol=0)
- Not NaN or infinity inputs: deterministic pass-through for positive values, zero for negative
