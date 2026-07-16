# Full Profiler Summary — layout_reduce_v1

## Profiler Configuration

- **Method**: torch.npu.Event (kernel-only latency via ACL events)
- **Warmup**: 200 iterations
- **Loops**: 100 iterations per repeat  
- **Repeat**: 3 (reduced from 5 due to total profiling time exceeding 4hrs for full msprof)
- **Measurement**: Primary = primary_compute_kernel equivalent (single kernel duration)
- **Note**: msprof profiling deferred to archive phase; torch.npu.Event consistent across all measurements

## Expand

| Batch | Torch (us) | Ascend C (us) | PyPTO (us) |
|-------|-----------|---------------|------------|
| 1 | 11.2 | 16.9 | 29400 |
| 2 | 12.8 | 17.0 | 58200 |
| 4 | 14.5 | 18.6 | 116800 |
| 8 | 17.3 | 28.5 | 233900 |
| 16 | 23.6 | 51.2 | 468100 |
| 32 | 36.1 | 92.5 | 937600 |
| 64 | 61.8 | 176.5 | 1870000 |

PyPTO: total device kernel aggregation for full output (B×256 per-row calls × ~3 kernels per row).

## Transpose

| Batch | Torch (us) | Ascend C (us) | PyPTO |
|-------|-----------|---------------|-------|
| 1 | 23.5 | 105.4 | BLOCKED |
| 2 | 27.8 | 198.1 | BLOCKED |
| 4 | 36.9 | 383.7 | BLOCKED |
| 8 | 56.1 | 759.5 | BLOCKED |
| 16 | 92.4 | 1509.4 | BLOCKED |
| 32 | 161.7 | 3008.8 | BLOCKED |
| 64 | 314.8 | 6001.4 | BLOCKED |

## ReduceSum

| Batch | Torch (us) | Ascend C (us) | PyPTO (us) |
|-------|-----------|---------------|------------|
| 1 | 27.8 | 20.8 | 118.2 |
| 2 | 29.5 | 19.8 | 119.5 |
| 4 | 31.2 | 31.0 | 117.8 |
| 8 | 35.6 | 31.2 | 120.3 |
| 16 | 44.1 | 54.0 | 125.6 |
| 32 | 60.2 | 100.1 | 132.4 |
| 64 | 95.7 | 191.9 | 148.1 |
