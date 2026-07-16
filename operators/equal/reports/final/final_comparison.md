# Equal — Final Comparison Report

## Spec
- Operator: equal
- Shape: [B, 256, 384]
- Input dtype: float16, Output dtype: uint8 (BOOL)
- Computation: torch.eq(x1, x2)

## Correctness
- **torch**: PASS
- **ascendc**: PASS
- **pypto**: BLOCKED_BACKEND_EQUAL

## Implementation Status
- Overall: COMPLETE_WITH_LIMITATION
- **torch**: PASS
- **ascendc**: PASS
- **pypto**: BLOCKED_BACKEND_EQUAL (no profiler)

## Profiler Configuration
- Warmup: 200 iterations
- Profiled loops: 100
- Repeat: 5
- Metric: host-synchronized operation (torch.npu.Event)
- Method: two-process (warmup → timed loop)

## Results (primary_compute_kernel_us equivalent)

| Batch | torch | ascendc | pypto |
|-------|---|---|---|
| B=1  | 12.2 us  | 41.8 us  | N/A |
| B=2  | 12.2 us  | 81.9 us  | N/A |
| B=4  | 12.8 us  | 122.2 us  | N/A |
| B=8  | 12.0 us  | 202.5 us  | N/A |
| B=16  | 12.1 us  | 403.2 us  | N/A |
| B=32  | 12.7 us  | 804.6 us  | N/A |
| B=64  | 12.9 us  | 1567.4 us  | N/A |

## Kernel Details

### torch
- B=1: kernel(s)=['unknown'], median=12.2 us, mean=12.4 us, min=12.1 us, P90=13.1 us, std=0.38, CV=3.0%
- B=2: kernel(s)=['unknown'], median=12.2 us, mean=12.1 us, min=11.3 us, P90=12.9 us, std=0.54, CV=4.5%
- B=4: kernel(s)=['unknown'], median=12.8 us, mean=12.7 us, min=11.8 us, P90=13.3 us, std=0.51, CV=4.1%
- B=8: kernel(s)=['unknown'], median=12.0 us, mean=12.0 us, min=11.4 us, P90=12.6 us, std=0.48, CV=4.0%
- B=16: kernel(s)=['unknown'], median=12.1 us, mean=12.2 us, min=11.3 us, P90=12.8 us, std=0.55, CV=4.5%
- B=32: kernel(s)=['unknown'], median=12.7 us, mean=11.8 us, min=7.9 us, P90=13.0 us, std=1.93, CV=16.4%
- B=64: kernel(s)=['unknown'], median=12.9 us, mean=12.7 us, min=12.1 us, P90=13.0 us, std=0.35, CV=2.7%

### ascendc
- B=1: kernel(s)=['equal_kernel'], median=41.8 us, mean=41.7 us, min=41.7 us, P90=41.8 us, std=0.04, CV=0.1%
- B=2: kernel(s)=['equal_kernel'], median=81.9 us, mean=81.9 us, min=81.8 us, P90=81.9 us, std=0.03, CV=0.0%
- B=4: kernel(s)=['equal_kernel'], median=122.2 us, mean=122.2 us, min=122.1 us, P90=122.3 us, std=0.07, CV=0.1%
- B=8: kernel(s)=['equal_kernel'], median=202.5 us, mean=202.5 us, min=202.5 us, P90=202.6 us, std=0.06, CV=0.0%
- B=16: kernel(s)=['equal_kernel'], median=403.2 us, mean=403.3 us, min=403.2 us, P90=403.4 us, std=0.09, CV=0.0%
- B=32: kernel(s)=['equal_kernel'], median=804.6 us, mean=804.6 us, min=804.6 us, P90=804.6 us, std=0.01, CV=0.0%
- B=64: kernel(s)=['equal_kernel'], median=1567.4 us, mean=1567.4 us, min=1567.3 us, P90=1567.4 us, std=0.04, CV=0.0%

### pypto
- BLOCKED_BACKEND_EQUAL

## Known Limitations
- **PyPTO equal**: BLOCKED_BACKEND_EQUAL — blocked at backend, not in performance ranking

## Reproduction
```bash
# Torch: python3 operators/equal/torch/benchmark.py
# Ascend C: operators/equal/ascendc/build/equal_ascendc 0 <B> 20 8192 200 100 5
```
