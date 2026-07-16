# Div Operator Specification (PyPTO)

## Operator
- Name: div
- Formula: Y[b,c1,c2,i,j] = X1[b,c1,c2,i,j] / X2[b,c1,c2,i,0]
- Category: broadcast arithmetic
- Description: Broadcast division along last dimension

## Inputs (Kernel Shapes)
- X1: [B, 12, 256, 256], float16, ND contiguous
- X2: [B, 12, 256, 1], float16, ND contiguous

## Output (Kernel Shape)
- Y: [B, 12, 256, 256], float16, ND contiguous

## Logical Shapes
- X1: [B, 3, 4, 256, 256], float16
- X2: [B, 3, 4, 256, 1], float16
- Y: [B, 3, 4, 256, 256], float16
- Dimension merge: (3,4) -> 12, contiguous zero-copy view, no actual reshape data movement

## Broadcast
- Axis: Last dimension (index 4 logical, index 3 kernel)
- Factor: 256 (each X2 scalar used 256 times)
- X2 must NOT be expanded to full X1 shape in GM

## Batches
B ∈ {1, 2, 4, 8, 16, 32} (B=64 optional)

## Precision
- RTOL: 0.001, ATOL: 0.001
- require_bitwise: false
- Note: FP16 Div cannot require bitwise equality due to division rounding

## Seed
20260715

## X1 Range
[-4, 4]

## X2 Range
[-4, -0.25] ∪ [0.25, 4] (no zero or near-zero in performance data)
