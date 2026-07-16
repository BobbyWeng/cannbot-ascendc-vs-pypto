# Mul Operator Specification

## Operator
- Name: mul
- Formula: Y = X1 * X2
- Category: element-wise arithmetic

## Inputs
- X1: [B, 3, 4, 256, 32], float16, ND contiguous
- X2: [B, 3, 4, 256, 32], float16, ND contiguous

## Output
- Y: [B, 3, 4, 256, 32], float16, ND contiguous

## Batches
B ∈ {1, 2, 4, 8, 16, 32, 64}

## Precision
- RTOL: 0, ATOL: 0, require_bitwise: true
- No signed-zero exemption (unlike ReLU)

## Seed
20260715
