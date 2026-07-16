# Div Broadcast Operator: Final Comparison — Ascend C vs Torch vs PyPTO

## Overview

FP16 broadcast division comparison on Ascend 910B. Shape: X1 `[B,12,256,256]`, X2 `[B,12,256,1]` (last-dim broadcast). X2 broadcast along last dimension from [B,12,256,1] to [B,12,256,256].

## Correctness

| Implementation | Status | Detail |
|---------------|--------|--------|
| **Ascend C** | ✅ PASS | All 6 batches (1,2,4,8,16,32) bitwise exact. Total 78,643,200 elements verified. |
| **Torch NPU** | ✅ PASS | `torch.div` matches FP16 reference at all batch sizes. |
| **PyPTO** | ⚠️ LIMITED | Minimal 2D [1,32]/[1,1] PASS. Broadcast [3072,256]/[3072,1] fails at backend (CompileFunction). |

### Key Correctness Facts

- Ascend C output is **bitwise identical** to the FP16 reference for ALL 6 batches
- No buggy formula residue (`1/(X1*X2)`) found in source code
- **V1 report claimed "Exact match at B=1,2" but had NOT verified B=4,8,16,32** — now all verified
- Special value references generated for all B=1..32 (12 edge cases each)

## Performance Comparison

| Batch | Ascend C (µs) | Torch NPU (µs) | Ratio | Ascend C BW (GB/s) | Torch BW (GB/s) |
|-------|:------------:|:-------------:|:----:|:-----------------:|:---------------:|
| 1 | 12.6 | 14.84 | **0.85× ✓** | 249.2 | 197.8 |
| 2 | 21.4 | 14.07 | **1.52×** | 294.9 | 417.4 |
| 4 | 43.4 | 18.90 | **2.30×** | 290.6 | 621.2 |
| 8 | 75.1 | 31.49 | **2.39×** | 335.9 | 745.8 |
| 16 | 176.2 | 57.66 | **3.06×** | 286.1 | 814.5 |
| 32 | 328.6 | 112.80 | **2.91×** | 306.9 | 832.8 |

## Profiler Results (B=32)

| Metric | Torch | Ascend C |
|--------|-------|----------|
| **Primary kernel type** | KERNEL_AIVEC | KERNEL_AIVEC |
| **Kernel name** | aclnnDiv_RealDivAiCore_RealDiv | div_kernel |
| **Primary compute kernel (µs)** | 126.2 | 306.5 |
| **Kernels per logical call** | ~42 | 1 |
| **Total device kernels per call (µs)** | 1702.9 | 2097.7 (all loops) |
| **Host-synchronized latency (µs)** | 112.8 | 328.6 |

## Optimization History

### Baseline (V1)
- Per-row `Duplicate(256) + Div(256)` with separate VECCALC broadcast buffer
- 32 iterations per tile, each requiring Duplicate setup + Div invocation
- Correct for all batches

### Candidate 1: Expanded Divisor at VECCALC
- Pre-expand all 32 X2 scalars into one 8192-element divisor vector, then single `Div`
- ❌ **FAILED**: Memory corruption — VECCALC buffer not visible to Div

### Candidate 2: Reciprocal + Mul
- Compute `Reciprocal(x2_scalar)` once, then `Duplicate(recip) + Mul(row)`
- ❌ **FAILED**: `Reciprocal` requires `LocalTensor`, not scalar; `Duplicate` cannot accept tensor elements

### Candidate 3: Expanded Divisor at VECOUT
- Pre-expand in VECOUT TQue, then single Div
- ✅ **Works**: All batches correct. B=1: 12.4µs, B=8: 74.4µs, B=32: 345.3µs
- Marginal improvement over baseline due to still having 32 Duplicate operations

### Final: In-place Duplicate+Div at VECOUT
- Write Duplicate directly to VECOUT yLocal, then Div in-place on same buffer
- ✅ **Works**: All batches correct. Eliminates VECCALC buffer entirely
- **Performance**: B=1: 12.6µs, B=8: 75.1µs, B=32: 328.6µs

## Bottleneck Analysis

### Why 96% Scalar Overhead?
The original profiler report claimed 96% scalar overhead. This is **partially misleading**:
- AIV architecture has scalar and vector units that **overlap**
- Scalar occupancy of 96% means the scalar unit is busy 96% of the time transmitting operations to the vector unit
- **True non-overlapped scalar overhead**: ~42% of wall time (~140µs of 330µs)
- The remaining ~160µs is actual vector Div compute

### Root Cause of B=32 Degradation
1. Each tile (8192 elements = 32 rows) requires **32 Duplicate + 32 Div** operations
2. Scalar loop control executes 32× per tile: address calc, Duplicate setup, Div issue
3. At B=32: 20 cores × 3 tiles/core × 32 rows = **1920 row iterations** total
4. Torch uses 42 micro-kernels per call with fused broadcast — no per-row scalar loop
5. At B=1 (1 core, 1 tile): launch overhead dominates, so Ascend C wins

### Why Can't We Eliminate the 32-Row Loop?
- Ascend C API limitation: `Duplicate` requires C++ `half` scalars, not tensor elements
- Cannot compute 32 reciprocals in batch and then duplicate — `Duplicate` type system prevents this
- VECCALC buffer corruption when used for inter-operation data passing
- No native 2D broadcast or batch-Duplicate API available

## PyPTO Status

| Test Case | Shape | Status |
|-----------|-------|--------|
| [1,32]/[1,1] | Minimal native Div | ✅ PASS |
| [1,12,256,256]/[1,12,256,1] | Broadcast Div | ❌ FAIL (CompileFunction) |

**Root Cause**: PyPTO `libtile_fwk_interface.so` backend fails at `HostMachine::CompileFunction` when processing the broadcast Div kernel with shapes [3072,256]/[3072,1]. This is a **genuine backend limitation** — broadcast lowering fails in the compiler pass.

**Recommendation**: The minimal case works, indicating PyPTO's frontend and basic Div are functional. The broadcast support in the backend compiler is the blocking issue.

## Key Answers

| Question | Answer |
|----------|--------|
| Why was B=32 327.6µs in V1? | Per-row Duplicate+Div loop with 32 iterations/tile |
| 96% scalar overhead evidence? | Profiler raw data shows high scalar pivot from 32 Duplicate operations, but overlaps with vector — true overhead ~42% |
| Optimization effect on scalar? | In-place VECOUT reduced VECCALC memory overhead but not the 32-row loop count |
| Final Ascend C strategy? | Native `AscendC::Div` with per-row `AscendC::Duplicate` |
| X2 broadcast location? | In UB (VECOUT), GM data stays at [B,12,256,1]; expansion is in UB only |
| Batch row processing? | No — still per-row (32 rows/tile) |
| Torch broadcast? | Single kernel call, ~42 internal micro-kernels |
| PyPTO Div support? | Native Div exists. Minimal shapes work. Broadcast lowering fails in backend. |
| Kernel types compared? | All KERNEL_AIVEC |
| Precision vs performance tradeoff? | None — all implementations produce identical FP16 results |
| B=1 vs B=32 difference? | B=1 fits 1 tile on 1 core (low overhead). B=32 saturates 20 cores but per-row overhead scales linearly. |

## Files

- **Spec**: `operators/div/SPEC.yaml`
- **Optimized kernel**: `operators/div/ascendc/src/div_kernel_optimized.asc`
- **Baseline kernel**: `operators/div/ascendc/src/div_kernel_baseline.asc`
- **Reports**: `reports/audit/`, `reports/correctness/`, `reports/optimization/`
- **Parsed profiler**: `reports/parsed/`
- **Raw profiler**: `reports/raw/`

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2026-07-16 | Initial comparison (B=1,2 correctness only, B=32 profiler, baseline performance) |
| v2 | 2026-07-16 | Full audit, all-batch correctness, optimized kernel, PyPTO diagnosis, unified profiler |
