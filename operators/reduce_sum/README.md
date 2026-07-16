# ReduceSum — Ascend C vs Torch vs PyPTO Comparison

## Operator
`Y[b,i] = sum_j X[b,i,j]` — reduce sum over last dimension (384), FP16

## Shapes
- X: `[B, 256, 384]`, FP16
- Y: `[B, 256]`, FP16
- Batches: B ∈ {1, 2, 4, 8, 16, 32, 64}

## Status: INCOMPLETE

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| Torch NPU | ❓ PENDING | Run `python3 torch/correctness.py` |
| Ascend C | ❓ PENDING | Build and run correctness |
| PyPTO | ❓ PENDING | Run `python3 pypto/correctness.py` |

## Input Coverage
- random_finite, all_zero, all_one, pos_neg_cancel, small_values, large_values, overflow_risk, underflow_risk, nan, inf

## Performance (median kernel latency, µs)

| B | Torch | Ascend C | PyPTO |
|---|:-----:|:--------:|:-----:|
| 1 | ❓ | ❓ | ❓ |
| 2 | ❓ | ❓ | ❓ |
| 4 | ❓ | ❓ | ❓ |
| 8 | ❓ | ❓ | ❓ |
| 16 | ❓ | ❓ | ❓ |
| 32 | ❓ | ❓ | ❓ |
| 64 | ❓ | ❓ | ❓ |

## Accumulation Dtype

| Implementation | Accumulation Dtype | Notes |
|--------------|-------------------|-------|
| torch.sum | FP32 (internal) | Standard PyTorch behavior |
| Ascend C | FP32 | Manual FP32 accum per row |
| PyPTO | TBD | Depends on backend implementation |

## Key Files
- `ascendc/src/reduce_sum_kernel.asc` — Ascend C kernel
- `pypto/src/reduce_sum_impl.py` — PyPTO implementation
- `data/generation_scripts/generate_inputs.py` — Data generation

See `reports/final/final_comparison.md` for full analysis.
