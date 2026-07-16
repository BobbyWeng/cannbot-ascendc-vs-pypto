# Or — Final Comparison Report

## Spec
- Operator: or
- Shape: [B, 256, 384]
- Input/Output dtype: uint8 (0/1 normalized)
- Computation: torch.logical_or(x1, x2)

## Correctness
- **torch**: PASS
- **ascendc**: PASS
- **pypto**: PASS

## Implementation Status
- Overall: COMPLETE
- **torch**: PASS
- **ascendc**: PASS
- **pypto**: PASS

## Profiler Configuration
- Warmup: 200 iterations
- Profiled loops: 100
- Repeat: 5
- Metric: host-synchronized operation (torch.npu.Event)
- Method: two-process (warmup → timed loop)

## Results (primary_compute_kernel_us equivalent)

| Batch | torch | ascendc | pypto |
|-------|---|---|---|
| B=1  | 256.3 us  | 6.5 us  | 148.8 us |
| B=2  | 315.0 us  | 6.6 us  | 149.1 us |
| B=4  | 434.1 us  | 7.9 us  | 142.3 us |
| B=8  | 748.6 us  | 7.9 us  | 148.4 us |
| B=16  | 1139.4 us  | 6.4 us  | 141.1 us |
| B=32  | 2072.8 us  | 7.2 us  | 146.1 us |
| B=64  | 4587.9 us  | 12.2 us  | 146.5 us |

## Kernel Details

### torch
- B=1: kernel(s)=['unknown'], median=256.3 us, mean=256.5 us, min=254.0 us, P90=258.5 us, std=1.60, CV=0.6%
- B=2: kernel(s)=['unknown'], median=315.0 us, mean=315.0 us, min=312.9 us, P90=316.6 us, std=1.29, CV=0.4%
- B=4: kernel(s)=['unknown'], median=434.1 us, mean=433.5 us, min=431.6 us, P90=435.0 us, std=1.39, CV=0.3%
- B=8: kernel(s)=['unknown'], median=748.6 us, mean=747.7 us, min=744.8 us, P90=749.0 us, std=1.60, CV=0.2%
- B=16: kernel(s)=['unknown'], median=1139.4 us, mean=1140.4 us, min=1138.8 us, P90=1144.0 us, std=1.88, CV=0.2%
- B=32: kernel(s)=['unknown'], median=2072.8 us, mean=2071.1 us, min=2063.2 us, P90=2075.9 us, std=4.27, CV=0.2%
- B=64: kernel(s)=['unknown'], median=4587.9 us, mean=4594.0 us, min=4573.5 us, P90=4631.8 us, std=19.77, CV=0.4%

### ascendc
- B=1: kernel(s)=['or_kernel'], median=6.5 us, mean=7.1 us, min=6.4 us, P90=8.1 us, std=0.75, CV=10.6%
- B=2: kernel(s)=['or_kernel'], median=6.6 us, mean=7.1 us, min=6.5 us, P90=8.1 us, std=0.75, CV=10.6%
- B=4: kernel(s)=['or_kernel'], median=7.9 us, mean=7.9 us, min=7.8 us, P90=8.1 us, std=0.11, CV=1.4%
- B=8: kernel(s)=['or_kernel'], median=7.9 us, mean=7.9 us, min=7.9 us, P90=8.0 us, std=0.05, CV=0.6%
- B=16: kernel(s)=['or_kernel'], median=6.4 us, mean=6.4 us, min=6.4 us, P90=6.6 us, std=0.08, CV=1.2%
- B=32: kernel(s)=['or_kernel'], median=7.2 us, mean=7.2 us, min=7.2 us, P90=7.3 us, std=0.00, CV=0.0%
- B=64: kernel(s)=['or_kernel'], median=12.2 us, mean=12.2 us, min=12.2 us, P90=12.2 us, std=0.03, CV=0.2%

### pypto
- B=1: kernel(s)=['unknown'], median=148.8 us, mean=149.4 us, min=148.5 us, P90=151.1 us, std=0.96, CV=0.6%
- B=2: kernel(s)=['unknown'], median=149.1 us, mean=148.7 us, min=147.0 us, P90=149.8 us, std=1.08, CV=0.7%
- B=4: kernel(s)=['unknown'], median=142.3 us, mean=142.4 us, min=141.2 us, P90=143.9 us, std=0.88, CV=0.6%
- B=8: kernel(s)=['unknown'], median=148.4 us, mean=148.5 us, min=148.3 us, P90=148.9 us, std=0.20, CV=0.1%
- B=16: kernel(s)=['unknown'], median=141.1 us, mean=141.1 us, min=139.7 us, P90=142.7 us, std=0.97, CV=0.7%
- B=32: kernel(s)=['unknown'], median=146.1 us, mean=146.5 us, min=144.1 us, P90=149.1 us, std=1.67, CV=1.1%
- B=64: kernel(s)=['unknown'], median=146.5 us, mean=146.7 us, min=145.9 us, P90=148.0 us, std=0.83, CV=0.6%

## Known Limitations
- PyPTO uses bitwise_or on 0/1 normalized uint8. For non-0/1 inputs, results differ from logical_or.

## Reproduction
```bash
# Torch: python3 operators/or/torch/benchmark.py
# Ascend C: operators/or/ascendc/build/or_ascendc 0 <B> 20 8192 200 100 5
# PyPTO: python3 operators/or/pypto/tests/test_or.py
```
