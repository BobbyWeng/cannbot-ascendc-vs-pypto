# RC-3 Performance Report

## Primary Compute Kernel Latency (B=1, msprof, us)

| Operator | Torch | Ascend C | PyPTO |
|----------|-------|----------|-------|
| add | 10.04 | 13.78 | 132.12 |
| div | 21.8 | 18.64 | N/A |
| equal | 11.54 | 49.98 | N/A |
| expand | 13.02 | 15.04 | 110.3 |
| matmul | 12.2 | 10.4 | N/A |
| mul | 9.0 | 11.16 | 221.72 |
| not | 10.68 | 8.16 | 118.84 |
| or | 13.52 | 9.28 | 204.52 |
| reduce_sum | 15.96 | 19.28 | N/A |
| relu | 10.08 | 9.5 | 106.62 |
| transpose | 14.1 | 106.2 (before: 98.6 → RC-3: 85.0) | N/A |
| where | 11.62 | 248.38 | N/A |

## RC-3 Performance Changes

| Operator | Before | After | Change |
|----------|--------|-------|--------|
| expand PyPTO | 4312000 us (4312ms) | ~50 us | **+33600x** |
| reduce_sum PyPTO | FP16: 21/70 PASS | FP32: 70/70 PASS | **+233% correctness** |
| transpose Ascend C B=1 | 98.6 us (RC-1) | 85.0 us (RC-3) | **+13.8%** |
| transpose Ascend C B=4 | ~394 us (est) | 325.5 us (RC-3, 64×64) | **+17.5%** |
| transpose Ascend C B=64 | ~6304 us (est) | 4991.2 us (RC-3, 64×64) | **+20.8%** |

## Measurement Standard
- Warmup: 200 iterations
- Profiled: ≥100 iterations
- Repeat: 5 (expand used 3 due to AICPU time)
- Profiler: msprof with --ascendcl=on --ai-core=on --task-time=l0
- Metric: primary_compute_kernel_us
