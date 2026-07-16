# Where — Final Comparison Report

## Spec
- Operator: where
- Shape: [B, 256, 384]
- Input: condition uint8, x1/x2 float16, Output: float16
- Computation: torch.where(condition, x1, x2)

## Correctness
- **torch**: PASS
- **ascendc**: PASS
- **pypto**: BLOCKED_BACKEND_WHERE_SELECT

## Implementation Status
- Overall: COMPLETE_WITH_LIMITATION
- **torch**: PASS
- **ascendc**: PASS
- **pypto**: BLOCKED_BACKEND_WHERE_SELECT (no profiler)

## Profiler Configuration
- Warmup: 200 iterations
- Profiled loops: 100
- Repeat: 5
- Metric: host-synchronized operation (torch.npu.Event)
- Method: two-process (warmup → timed loop)

## Results (primary_compute_kernel_us equivalent)

| Batch | torch | ascendc | pypto |
|-------|---|---|---|
| B=1  | 131.9 us  | 238.6 us  | N/A |
| B=2  | 159.1 us  | 475.7 us  | N/A |
| B=4  | 230.9 us  | 712.7 us  | N/A |
| B=8  | 350.2 us  | 1186.9 us  | N/A |
| B=16  | 578.3 us  | 2372.2 us  | N/A |
| B=32  | 1010.9 us  | 4742.3 us  | N/A |
| B=64  | 2180.7 us  | 9245.7 us  | N/A |

## Kernel Details

### torch
- B=1: kernel(s)=['unknown'], median=131.9 us, mean=131.8 us, min=131.0 us, P90=132.6 us, std=0.55, CV=0.4%
- B=2: kernel(s)=['unknown'], median=159.1 us, mean=159.3 us, min=158.1 us, P90=160.8 us, std=1.06, CV=0.7%
- B=4: kernel(s)=['unknown'], median=230.9 us, mean=230.9 us, min=229.4 us, P90=232.9 us, std=1.16, CV=0.5%
- B=8: kernel(s)=['unknown'], median=350.2 us, mean=350.4 us, min=347.0 us, P90=353.8 us, std=2.17, CV=0.6%
- B=16: kernel(s)=['unknown'], median=578.3 us, mean=581.8 us, min=573.8 us, P90=599.9 us, std=9.42, CV=1.6%
- B=32: kernel(s)=['unknown'], median=1010.9 us, mean=1011.7 us, min=1007.2 us, P90=1018.0 us, std=3.61, CV=0.4%
- B=64: kernel(s)=['unknown'], median=2180.7 us, mean=2187.0 us, min=2152.8 us, P90=2239.8 us, std=29.28, CV=1.3%

### ascendc
- B=1: kernel(s)=['where_kernel'], median=238.6 us, mean=238.6 us, min=238.6 us, P90=238.6 us, std=0.01, CV=0.0%
- B=2: kernel(s)=['where_kernel'], median=475.7 us, mean=475.7 us, min=475.6 us, P90=475.8 us, std=0.06, CV=0.0%
- B=4: kernel(s)=['where_kernel'], median=712.7 us, mean=712.7 us, min=712.7 us, P90=712.8 us, std=0.01, CV=0.0%
- B=8: kernel(s)=['where_kernel'], median=1186.9 us, mean=1187.0 us, min=1186.9 us, P90=1187.1 us, std=0.10, CV=0.0%
- B=16: kernel(s)=['where_kernel'], median=2372.2 us, mean=2372.1 us, min=2372.0 us, P90=2372.3 us, std=0.09, CV=0.0%
- B=32: kernel(s)=['where_kernel'], median=4742.3 us, mean=4742.4 us, min=4742.2 us, P90=4742.5 us, std=0.09, CV=0.0%
- B=64: kernel(s)=['where_kernel'], median=9245.7 us, mean=9245.7 us, min=9245.7 us, P90=9245.8 us, std=0.08, CV=0.0%

### pypto
- BLOCKED_BACKEND_WHERE_SELECT

## Known Limitations
- **PyPTO where**: BLOCKED_BACKEND_WHERE_SELECT — blocked at backend, not in performance ranking

## Reproduction
```bash
# Torch: python3 operators/where/torch/benchmark.py
# Ascend C: operators/where/ascendc/build/where_ascendc 0 <B> 20 8192 200 100 5
```
