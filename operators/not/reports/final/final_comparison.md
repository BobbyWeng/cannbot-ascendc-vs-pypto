# Not — Final Comparison Report

## Spec
- Operator: not
- Shape: [B, 256, 384]
- Input/Output dtype: uint8
- Computation: torch.logical_not(x)

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
| B=1  | 127.5 us  | 6.4 us  | 136.6 us |
| B=2  | 160.6 us  | 7.6 us  | 133.2 us |
| B=4  | 220.5 us  | 7.7 us  | 135.1 us |
| B=8  | 358.4 us  | 6.5 us  | 136.5 us |
| B=16  | 563.8 us  | 6.4 us  | 130.7 us |
| B=32  | 1028.6 us  | 7.9 us  | 132.5 us |
| B=64  | 1986.7 us  | 12.0 us  | 129.9 us |

## Kernel Details

### torch
- B=1: kernel(s)=['unknown'], median=127.5 us, mean=128.0 us, min=127.0 us, P90=129.6 us, std=0.98, CV=0.8%
- B=2: kernel(s)=['unknown'], median=160.6 us, mean=160.7 us, min=160.1 us, P90=161.9 us, std=0.63, CV=0.4%
- B=4: kernel(s)=['unknown'], median=220.5 us, mean=220.6 us, min=217.7 us, P90=224.9 us, std=2.39, CV=1.1%
- B=8: kernel(s)=['unknown'], median=358.4 us, mean=358.3 us, min=356.1 us, P90=361.2 us, std=1.92, CV=0.5%
- B=16: kernel(s)=['unknown'], median=563.8 us, mean=564.5 us, min=563.0 us, P90=566.9 us, std=1.42, CV=0.3%
- B=32: kernel(s)=['unknown'], median=1028.6 us, mean=1027.5 us, min=1022.5 us, P90=1031.2 us, std=2.90, CV=0.3%
- B=64: kernel(s)=['unknown'], median=1986.7 us, mean=1989.2 us, min=1986.1 us, P90=1996.6 us, std=3.99, CV=0.2%

### ascendc
- B=1: kernel(s)=['not_kernel'], median=6.4 us, mean=6.5 us, min=6.3 us, P90=7.0 us, std=0.26, CV=4.0%
- B=2: kernel(s)=['not_kernel'], median=7.6 us, mean=7.7 us, min=7.6 us, P90=7.9 us, std=0.12, CV=1.6%
- B=4: kernel(s)=['not_kernel'], median=7.7 us, mean=7.2 us, min=6.3 us, P90=7.8 us, std=0.64, CV=8.9%
- B=8: kernel(s)=['not_kernel'], median=6.5 us, mean=6.5 us, min=6.3 us, P90=6.6 us, std=0.11, CV=1.7%
- B=16: kernel(s)=['not_kernel'], median=6.4 us, mean=6.5 us, min=6.4 us, P90=6.6 us, std=0.05, CV=0.8%
- B=32: kernel(s)=['not_kernel'], median=7.9 us, mean=7.6 us, min=7.2 us, P90=8.1 us, std=0.39, CV=5.1%
- B=64: kernel(s)=['not_kernel'], median=12.0 us, mean=12.0 us, min=12.0 us, P90=12.0 us, std=0.01, CV=0.1%

### pypto
- B=1: kernel(s)=['unknown'], median=136.6 us, mean=136.8 us, min=134.9 us, P90=139.5 us, std=1.71, CV=1.3%
- B=2: kernel(s)=['unknown'], median=133.2 us, mean=133.9 us, min=132.4 us, P90=136.6 us, std=1.48, CV=1.1%
- B=4: kernel(s)=['unknown'], median=135.1 us, mean=135.4 us, min=134.6 us, P90=137.1 us, std=0.89, CV=0.7%
- B=8: kernel(s)=['unknown'], median=136.5 us, mean=136.7 us, min=135.5 us, P90=138.0 us, std=1.03, CV=0.8%
- B=16: kernel(s)=['unknown'], median=130.7 us, mean=131.3 us, min=129.7 us, P90=134.2 us, std=1.56, CV=1.2%
- B=32: kernel(s)=['unknown'], median=132.5 us, mean=133.6 us, min=132.0 us, P90=136.7 us, std=1.84, CV=1.4%
- B=64: kernel(s)=['unknown'], median=129.9 us, mean=130.5 us, min=129.3 us, P90=132.4 us, std=1.18, CV=0.9%

## Known Limitations

## Reproduction
```bash
# Torch: python3 operators/not/torch/benchmark.py
# Ascend C: operators/not/ascendc/build/not_ascendc 0 <B> 20 8192 200 100 5
# PyPTO: python3 operators/not/pypto/tests/test_not.py
```
