# Div Broadcast — Ascend C vs Torch vs PyPTO

## Operator
`Y[b,c1,c2,i,j] = X1[b,c1,c2,i,j] / X2[b,c1,c2,i,0]`

Broadcast division along last dimension.

## Shapes
- **Logical**: X1/Y `[B,3,4,256,256]`, X2 `[B,3,4,256,1]`
- **Kernel**: X1/Y `[B,12,256,256]`, X2 `[B,12,256,1]`
- **Dimension merge**: `(3,4) -> 12`, contiguous zero-copy view

## Status: COMPLETE ✅

## Correctness
| Implementation | Status | Detail |
|---------------|--------|--------|
| Ascend C | ✅ PASS | All 6 batches bitwise exact with FP16 reference |
| Torch NPU | ✅ PASS | `torch.div` matches reference |
| PyPTO | ⚠️ LIMITED | Minimal shapes work; broadcast fails at backend |

## Performance (B=32)
| Implementation | Latency (µs) | vs Torch |
|---------------|:------------:|:--------:|
| **Ascend C (optimized)** | 328.6 | 2.91× |
| **Torch NPU** | 112.8 | 1.00× |
| **PyPTO** | N/A | Backend blocked |

## Implementation Strategies
| Implementation | Kernel Type | Broadcast Method |
|--------------|-------------|-----------------|
| torch.div | KERNEL_AIVEC (42 micro-kernels/call) | Vendor fused broadcast-div |
| Ascend C | KERNEL_AIVEC (1 kernel/call) | Per-row Duplicate+Div in UB |
| PyPTO | — | Backend CompileFunction fails |

## Key Files
- `ascendc/src/div_kernel.asc` — Current kernel (baseline)
- `ascendc/src/div_kernel_optimized.asc` — In-place VECOUT Duplicate+Div (best perf)
- `ascendc/src/div_kernel_baseline.asc` — Original baseline (V1) with VECCALC buffer
- `reports/final/final_comparison.md` — Full comparison report
- `reports/correctness/ascendc_all_correctness.json` — Correctness results
- `reports/audit/current_state.md` — Initial state audit

## Optimization Summary
- Candidate 1 (expanded divisor VECCALC): FAILED (memory corruption)
- Candidate 2 (Reciprocal+Mul): FAILED (scalar API limitation)
- Candidate 3 (expanded divisor VECOUT): PASS, ~1-11% improvement
- **Final (in-place Duplicate+Div VECOUT)**: PASS, ~6% improvement vs baseline at B=32

## Bottleneck
Fundamental 32-row Duplicate+Div loop per tile (8192 elements). Cannot eliminate without deeper vendor API support. Ascend C outperforms torch at B=1 but is 2.9× slower at B=32.
