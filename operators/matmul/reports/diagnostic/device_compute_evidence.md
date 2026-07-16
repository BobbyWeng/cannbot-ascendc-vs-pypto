# Device Compute Evidence

## Kernel Type

The MatMul kernel uses `MatmulImpl` from `adv_api/matmul/matmul.h`, which routes to the hardware Cube MMAD pipeline:

- Kernel entry: `matmul_kernel` with `__global__ __aicore__` annotation
- No `__vector__` — uses the high-level MatMul API that the compiler lowers to Cube instructions
- The `MatmulImpl::IterateAll()` call triggers: GM→L1→L0A/L0B→MMAD→L0C→Fixpipe→GM

## Expected Profiler Evidence

When profiled with msprof, we expect:
- Kernel type: `KERNEL_AIC_CUBE` or `KERNEL_MIX_AIC`
- Cube compute event duration matches the main compute time
- AIC active during compute (not AIV)

## Kernel Count per Call

Each single `matmul_kernel<<<>>>` invocation processes `matricesPerCore` matrices sequentially.
Each matrix uses one `MatmulImpl::IterateAll()` call.

- For B=1 (12 matrices), blockDim=20: 12 matrices ÷ 20 cores = 1 matrix/core, some cores idle
- For B=32 (384 matrices), blockDim=20: 384 ÷ 20 = 19.2, 19 matrices/core + 4 tail
