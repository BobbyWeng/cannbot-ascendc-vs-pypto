# Profiler Summary — layout_reduce_v1

## Profiler Status

| Operator | Torch | PyPTO | Ascend C |
|----------|-------|-------|----------|
| Expand | PENDING | PENDING | HOST_FALLBACK (not comparable) |
| Transpose | PENDING | BLOCKED_BACKEND (large shape) | HOST_FALLBACK (not comparable) |
| ReduceSum | PENDING | PENDING | HOST_FALLBACK (not comparable) |

## Detailed msprof Results

Full msprof profiling deferred to batch completion due to ~8hr sequential NPU time requirement. Results will be captured when the full operator batch pipeline runs.

## Expand Profiler Notes

When running, must use msprof with:
```
--ascendcl=on --ai-core=on --task-time=l0
```

### Torch Expand
- Call: `x.expand(B,256,384).contiguous()`
- Per-batch measurement: B=1..64 with warmup=200, loops=100, repeat=5
- Expected kernels: 1 MIX_AIC compute kernel per call

### PyPTO Expand
- Call: `expand_wrapper(x)` → B×256 per-row `expand_clone([1]→[384])` calls
- Each logical expand requires B×256 JIT/runtime dispatches
- Primary metric: `all_device_kernels_us_per_call` (sum of all per-row kernels for full output)
- NOT per-row kernel time
- Expected kernels per logical call: B×256 × (3 kernels per row: 1 MIX_AIC + 2 AICPU)
- First JIT compile cost excluded from main measurement

### Ascend C Expand
- Identity kernel only (Add+Sub = x+x-x)
- Host pre-expand excluded from device-kernel ranking
- Label: `HOST_FALLBACK_NOT_COMPARABLE`

## Transpose Profiler Notes

### Torch Transpose
- Call: `x.transpose(1,2).contiguous()`  
- Expected kernels: 1 MIX_AIC materialization kernel

### PyPTO Transpose
- Large shape [256,384]: **NOT PROFILED** (BLOCKED_BACKEND — CompileFunction pass failure)
- Small shape diagnostic [16,32]: can be profiled separately

### Ascend C Transpose
- Host pre-transpose + identity copy only
- Label: `HOST_FALLBACK_NOT_COMPARABLE`

## ReduceSum Profiler Notes

### Torch ReduceSum
- Call: `torch.sum(x, dim=-1)`
- Expected accumulation: FP32 internally, FP16 output
- Expected kernels: reduction kernel

### PyPTO ReduceSum
- Call: `reduce_sum_wrapper(x)` → `pypto.op.sum(x, dim=-1)`
- Accumulation: FP16
- Expected kernels: 1+ reduction kernels on device
- FP16 accumulation precision vs FP32 torch: documented in correctness

### Ascend C ReduceSum
- Host FP32 pre-reduce + identity copy only
- Label: `HOST_FALLBACK_NOT_COMPARABLE`
