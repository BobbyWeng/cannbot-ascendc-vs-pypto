# MatMul — Final Audit Report

## Status: COMPLETE

### Summary

| Implementation | Status | Correctness | Profiler | Notes |
|---------------|--------|-------------|----------|-------|
| **Torch** | COMPLETE | PASS (all 6 batches) | ✅ msprof | Reference baseline |
| **Ascend C** | COMPLETE | PASS (all 6 batches) | ✅ msprof + aclrtEvent | TRUE_CUBE_IMPLEMENTATION |
| **PyPTO** | COMPLETE_WITH_LIMITATION | N/A | N/A | BLOCKED_BACKEND |

## Audit Checklist

### ✓ Ascend C — TRUE Cube Kernel

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `__cube__` kernel annotation | ✅ | `matmul_kernel.asc:60` |
| Cube MMAD API (`MatmulImpl`) | ✅ | `matmul_kernel.asc:32-44` |
| No host fallback | ✅ | Direct `<<<>>>` launch |
| No ACLNN wrapper | ✅ | No `aclnnMatmul*` calls |
| Tiling via `MultiCoreMatmulTiling` | ✅ | `matmul_host.asc:90` |
| All batches B=1..32 run | ✅ | Verified on NPU |
| Correctness PASS | ✅ | All 6 batches (perf_fp16) |
| msprof data | ✅ | All 6 batches |

### ✓ Torch Baseline

| Criterion | Status |
|-----------|--------|
| Correctness PASS | ✅ |
| msprof data | ✅ |
| Host-synchronized latency | ✅ (torch.npu.Event) |

### ✓ PyPTO — Orchestrator

| Criterion | Status |
|-----------|--------|
| Stage 1 (SPEC) | ✅ |
| Stage 2 (API_REPORT) | ✅ |
| Stage 3 (golden) | ✅ |
| Stage 4 (DESIGN) | ✅ |
| Stage 5 (impl+test) | ✅ |
| Stage 6 (precision) | N/A (BLOCKED_BACKEND) |
| Stage 7 (perf) | N/A (BLOCKED_BACKEND) |
| DIAGNOSTIC_REPORT.md | ✅ |
| BLOCKED_BACKEND documented | ✅ |

### ✓ Benchmark

| Criterion | Status |
|-----------|--------|
| Warmup 200 | ✅ |
| Profiled loops 100 | ✅ |
| Repeat 5 | ✅ |
| msprof `--ascendcl=on --ai-core=on --task-time=l0` | ✅ |
| Primary compute kernel metric | ✅ |
| All device kernels metric | ✅ |

### ✓ Profiler

| Criterion | Status |
|-----------|--------|
| Raw data in `reports/raw/` | ✅ (Torch + Ascend C, B=1..32) |
| Parsed data in `reports/parsed/` | ✅ |
| Kernel names identified | ✅ |
| Kernel types classified | ✅ |
| PyPTO not profiled | ✅ (BLOCKED_BACKEND) |

### ✓ Reports

| Criterion | Status |
|-----------|--------|
| Final comparison report | ✅ (`reports/final/final_comparison.md`) |
| Ascend C audit report | ✅ (`reports/audit/ascendc_matmul_audit.md`) |

### ✓ Dashboard

| Criterion | Status |
|-----------|--------|
| `dashboard/dashboard.json` updated | ✅ |
| `reports/release/current_release.json` updated | ✅ |
| Operator status correct | ✅ |
| Known limitations documented | ✅ |

## Known Limitations

1. **Ascend C per-matrix dispatch**: Inefficient — 12-384 separate `<<<1, 0>>>` kernel launches per logical operation. Should be batched into a single call with proper multi-core dispatch.

2. **PyPTO BLOCKED_BACKEND**: The Cube tiling engine (FC4000: invalid tile values) blocks ALL matmul shapes. `pypto.matmul` is fundamentally broken in this version.

3. **Kernel type classification**: msprof reports both torch `aclnnMatmul` and Ascend C `matmul_kernel` as `KERNEL_AICORE` rather than `KERNEL_AIC_CUBE`. This is a profiler classification behavior.

4. **Only perf_fp16 correctness**: The Ascend C host binary only processes `perf_fp16` inputs. Special test cases (zeros, ones, nan, inf, etc.) are not generated.

## Final Status

**COMPLETE**

Torch and Ascend C both have full correctness and profiler data. PyPTO is BLOCKED_BACKEND with documented limitation.
