# Operator: Or (LogicalOr)

## Formula
Y = LogicalOr(X1, X2)

## Inputs
| Name | Shape | Dtype | Layout |
|------|-------|-------|--------|
| X1   | [B, 256, 384] | bool | ND contiguous |
| X2   | [B, 256, 384] | bool | ND contiguous |

## Output
| Name | Shape | Dtype | Layout |
|------|-------|-------|--------|
| Y    | [B, 256, 384] | bool | ND contiguous |

## Behavior
Element-wise logical OR on BOOL tensors.
- BOOL is stored as uint8 (0=False, 1=True).
- Any non-zero input value is treated as True.
- Output is 0 or 1 only.

## Batches
B ∈ {1, 2, 4, 8, 16, 32, 64}

## Precision
- require_bitwise: true
- atol: 0, rtol: 0
