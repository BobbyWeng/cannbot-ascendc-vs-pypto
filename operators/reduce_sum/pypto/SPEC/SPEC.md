# ReduceSum Operator Specification

## Formula
Y[b,i] = sum_j X[b,i,j] — reduction on last dimension (384)

## Inputs
- X: [B, 256, 384], float16, ND contiguous

## Outputs
- Y: [B, 256], float16, ND contiguous

## Batches
B ∈ {1, 2, 4, 8, 16, 32, 64}

## Precision
- rtol: 0.01, atol: 0.01
- require_bitwise: false
- FP16 reduction may have accumulation differences
- Two references: FP16-path (simulated FP16 chained accum) and FP32 accum (torch.sum)

## Shape Constraints
- axis=-1, keepdims=false
- Output shape is [B, 256]
