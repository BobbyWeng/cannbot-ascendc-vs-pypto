# Mul Comparison — Profiler-Based Device Kernel Analysis

## Experiment Environment

| Item | Value |
|------|-------|
| Operator | Mul (Y = X1 * X2) |
| Device | Ascend 910B (2 chips, device 0 measured) |
| dtype | FP16 |
| Shape | [B, 3, 4, 256, 32] |
| Batches | 1, 2, 4, 8, 16, 32, 64 |
| Warmup | 200 |
| Profiled loops | 100 |
| Repeat | 5 (torch/ascendc), 1 (pypto) |
| Profiler | msprof --ascendcl=on --ai-core=on --task-time=l0 |
| Seed | 20260715 |

## Correctness

| Implementation | Status | Note |
|--------------|--------|------|
| torch.mul | ✅ PASS | 7/7 batches strict bitwise (no signed-zero exemption) |
| Ascend C Mul | ✅ PASS | 7/7 batches strict bitwise (no signed-zero exemption) |
| PyPTO Mul | ✅ PASS | 7/7 batches strict bitwise (no signed-zero exemption) |

All three implementations produce **strict bitwise-identical** results across all 7 batch sizes,
including signed-zero cases (+0 × -0 = -0 per IEEE 754). No tolerance needed.

## Generation Methods

| Implementation | Method | Kernel Type | Kernels/logical call |
|--------------|--------|-------------|---------------------|
| torch.mul | Built-in `torch_npu.mul()` → internal aclnnMul | KERNEL_AIVEC | 1 |
| Ascend C | Custom vector kernel (`mul_kernel.asc`) via `<<<>>>` invoke | KERNEL_AIVEC | 1 |
| PyPTO | `@pypto.frontend.jit` with `pypto.op.mul(x1, x2)` → mul_kernel_2d | KERNEL_MIX_AIC + 2x KERNEL_AICPU | 3 (1 compute + 2 executor) |

## Profiler Method

| Implementation | Session type | JIT excluded? | First-call init excluded? |
|--------------|-------------|---------------|--------------------------|
| torch.mul | Single msprof: warmup(200)+loops(100)×5 | N/A | No — events include all |
| Ascend C | Single msprof: warmup(200)+loops(100)×5 | N/A | No — events include all |
| PyPTO | Two-process: warmup(200) no-profiler, then msprof(100) | ✅ Yes | ✅ Yes (one-time ~3ms AICPU init excluded) |

## PyPTO Kernel Event Analysis

PyPTO produces **3 kernel events per logical call**:

| Event # | Task Type | Name | Role |
|---------|-----------|------|------|
| 1 | KERNEL_AICPU | PYPTO_mul_kernel_2d | PyPTO runtime executor on AI CPU (tiling, dispatch) |
| 2 | KERNEL_AICPU | PYPTO_mul_kernel_2d | PyPTO runtime executor on AI CPU (tiling, dispatch) |
| 3 | KERNEL_MIX_AIC | PYPTO_mul_kernel_2d | AI Core compute kernel (actual Mul computation) |

## Table A — All Device Kernels Per Logical Call

**Metric**: `all_device_kernels_us_per_call` (mean of profiled loop events, per logical call)

| Batch | torch (AIVEC) | Ascend C (AIVEC) | PyPTO (MIX_AIC + 2×AICPU) | Ascend C vs torch | PyPTO vs Ascend C |
|-------|--------------|-----------------|---------------------------|-------------------|-------------------|
| 1 | 3.88 | 3.14 | 177.39 | 1.24x | 56.5x |
| 2 | 3.95 | 3.53 | 174.19 | 1.12x | 49.3x |
| 4 | 4.22 | 4.03 | 176.95 | 1.05x | 43.9x |
| 8 | 4.84 | 4.87 | 192.62 | 0.99x | 39.6x |
| 16 | 6.04 | 6.69 | 254.61 | 0.90x | 38.1x |
| 32 | 8.18 | 10.43 | 338.58 | 0.78x | 32.5x |
| 64 | 12.52 | 17.64 | 496.07 | 0.71x | 28.1x |

**Key observation**: Torch and Ascend C both use KERNEL_AIVEC with very similar performance.
PyPTO total (compute + executor) is 28-56x slower.

## Table B — PyPTO Breakdown (Per Logical Call)

| Batch | KERNEL_MIX_AIC compute (us) | 2x KERNEL_AICPU executor (us) | All device events sum (us) | Composition |
|-------|---------------------------|-----------------------------|--------------------------|-------------|
| 1 | 57.7 | 119.7 | 177.4 | 57.7 + 119.7 = 177.4 |
| 2 | 57.1 | 117.1 | 174.2 | 57.1 + 117.1 = 174.2 |
| 4 | 59.6 | 117.4 | 177.0 | 59.6 + 117.4 = 177.0 |
| 8 | 66.8 | 125.9 | 192.6 | 66.8 + 125.9 = 192.6 |
| 16 | 93.2 | 161.4 | 254.6 | 93.2 + 161.4 = 254.6 |
| 32 | 136.4 | 202.2 | 338.6 | 136.4 + 202.2 = 338.6 |
| 64 | 207.0 | 289.0 | 496.1 | 207.0 + 289.0 = 496.1 |

## Kernel Type Comparison

| Aspect | torch.mul | Ascend C Mul | PyPTO Mul |
|--------|----------|-------------|-----------|
| Kernel name | aclnnMul_MulAiCore_Mul | mul_kernel | PYPTO_mul_kernel_2d |
| Kernel type | KERNEL_AIVEC | KERNEL_AIVEC | KERNEL_MIX_AIC + KERNEL_AICPU |
| Kernels per call | 1 | 1 | 3 |
| Compute kernel type | KERNEL_AIVEC | KERNEL_AIVEC | KERNEL_MIX_AIC |
| Executor overhead | None | None | 2× KERNEL_AICPU |

## Key Differences from ReLU

| Aspect | ReLU | Mul |
|--------|------|-----|
| Torch kernel name | aclnnRelu_Relu_Relu | aclnnMul_MulAiCore_Mul |
| Torch kernel type | KERNEL_AIVEC | KERNEL_AIVEC |
| Ascend C kernel name | relu_kernel | mul_kernel |
| Ascend C kernel type | KERNEL_AIVEC | KERNEL_AIVEC |
| PyPTO kernel type | KERNEL_MIX_AIC + 2×KERNEL_AICPU | KERNEL_MIX_AIC + 2×KERNEL_AICPU |
| Signed-zero exemption | ✅ Yes (max(-0.0, +0.0) ambiguous) | ❌ No (Mul IEEE 754 well-defined) |
| Correctness standard | Bitwise (sz-exempted) | Strict bitwise |
| Inputs | 1 (x) | 2 (x1, x2) |
| Elements per batch | B × 98304 | B × 98304 |
| Bytes per element | 4 (read+write) | 6 (read x2 + write) |

## One-Time Init

PyPTO has a one-time ~3ms KERNEL_AICPU event on first call per process (cache load/runtime init).
This is NOT a per-call cost.

## Final Summary

| Metric | Value |
|--------|-------|
| Overall Status | PASS |
| Correctness | ✅ PASS (all 7 batches strict bitwise, including signed-zero) |
| Unified Device Profiler Comparison | ✅ PASS |
| torch.mul kernel | KERNEL_AIVEC ~3.9-12.5 us |
| Ascend C Mul kernel | KERNEL_AIVEC ~3.1-17.6 us |
| PyPTO compute kernel | KERNEL_MIX_AIC ~57.7-207.0 us |
| PyPTO executor overhead | 2× KERNEL_AICPU ~117-289 us additional |
| Fastest compute kernel | torch.mul KERNEL_AIVEC (~3.9 us at B=1) |
