# Cannbot: Ascend C vs PyPTO Operator Comparison

A structured, reproducible comparison framework for evaluating Ascend C and PyPTO operator implementations against torch baselines on Ascend NPU hardware.

## Project Structure

```
cannbot_ascendc_vs_pypto/
├── README.md                        # This file
├── AGENTS.md                        # Agent configuration
├── environment/                     # Environment and version manifests
├── common/                          # Shared libraries
├── operators/                       # Operator comparison directories
│   ├── relu/                        # ReLU (COMPLETE)
│   ├── add/                         # Addition (COMPLETE_WITH_LIMITATION)
│   ├── mul/                         # Multiplication (COMPLETE)
│   ├── div/                         # Division (COMPLETE_WITH_LIMITATION)
│   ├── equal/                       # Equal (COMPLETE_WITH_LIMITATION)
│   ├── not/                         # LogicalNot (COMPLETE_WITH_LIMITATION)
│   ├── or/                          # LogicalOr (COMPLETE_WITH_LIMITATION)
│   ├── where/                       # Where (COMPLETE_WITH_LIMITATION)
│   ├── expand/                      # Expand (PARTIAL)
│   ├── transpose/                   # Transpose (PARTIAL)
│   ├── reduce_sum/                  # ReduceSum (PARTIAL)
├── templates/                       # Templates for new operators
├── reports/                         # Project-level reports
│   ├── release/                     # Current release (single source of truth)
│   │   ├── current_release.json     # Machine-readable release state
│   │   ├── current_release.md       # Human-readable release summary
│   │   ├── operator_matrix.csv      # Operator status matrix
│   │   ├── performance_matrix.csv   # Performance comparison
│   │   ├── correctness_matrix.csv   # Correctness coverage
│   │   ├── limitation_matrix.md     # Known limitations
│   │   └── limitation_matrix.json   # Machine-readable limitations
│   ├── batches/                     # Batch-level current reports
│   └── operator_summary.md          # Quick-reference summary
├── scripts/                         # Project-level scripts
└── archives/                        # Operator archive packages
```

## Operator Status

| Operator | Final Status | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------------|-------|----------|-------|-------------|----------|
| relu | **COMPLETE** | ✅ PASS | ✅ TRUE_DEVICE | ✅ SUCCESS | ✅ Full batch | ✅ msprof |
| mul | **COMPLETE** | ✅ PASS | ✅ TRUE_DEVICE | ✅ SUCCESS | ✅ Full batch | ✅ msprof |
| add | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ✅ SUCCESS | ⚠️ PyPTO B=1 only persisted | ✅ msprof |
| div | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ❌ BLOCKED_BACKEND | ✅ Torch+AscendC | ✅ msprof |
| equal | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ❌ BLOCKED_BACKEND | ✅ Torch+AscendC | ⚠️ Event only |
| not | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ✅ SUCCESS | ❌ AscendC FAIL (script bug) | ⚠️ Event only |
| or | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ⚠️ bitwise_or | ❌ AscendC FAIL (script bug) | ⚠️ Event only |
| where | **COMPLETE_WITH_LIMITATION** | ✅ PASS | ✅ TRUE_DEVICE | ❌ BLOCKED_BACKEND | ✅ Torch+AscendC | ⚠️ Event only |
| expand | **PARTIAL** | ⚠️ B=1 only | ✅ TRUE_DEVICE (unverified) | ✅ PASS (dispatch) | ⚠️ Major gaps | ❌ No msprof |
| transpose | **PARTIAL** | ⚠️ B=1 only | ✅ TRUE_DEVICE (unverified) | ⚠️ Partial (small PASS) | ⚠️ Major gaps | ❌ No msprof |
| reduce_sum | **PARTIAL** | ⚠️ B=1 only | ✅ TRUE_DEVICE (unverified) | ✅ SUCCESS (unverified) | ⚠️ Major gaps | ❌ No msprof |

**Summary**: 2 COMPLETE, 6 COMPLETE_WITH_LIMITATION, 3 PARTIAL

## Key Corrections vs Previous Reports

- **Expand/Transpose/ReduceSum Ascend C**: Previous reports claimed `HOST_PRECOMPUTE_FALLBACK`. Source code audit confirms all three are **TRUE_DEVICE_IMPLEMENTATION** with genuine device-side kernels (Duplicate, tile-transpose, ReduceSum Level 2). However, correctness and profiler data have not been collected on hardware.
- **Not/Or Ascend C**: Ascend C correctness shows FAIL (missing reference_bool.bin files). Kernel may be correct — script filename pattern bug.
- **All logical operators**: Profiling uses `torch.npu.Event`/`aclrtEvent`, NOT msprof. These latencies are NOT comparable with arithmetic operator msprof data.

## Measurement Methodology

All arithmetic operators use **unified profiler-based measurement**:
1. **Profiler**: msprof with `--ascendcl=on --ai-core=on --task-time=l0`
2. **Warmup**: 200 iterations (excluded from measurement)
3. **Profiled iterations**: 100 iterations
4. **Key metric**: `primary_compute_kernel_us` — device kernel duration
5. **PyPTO**: Two-process method (warmup no-profiler, then msprof) to exclude JIT compilation

## Requirements

- Ascend 910B or compatible NPU
- CANN Toolkit (tested with 9.0.0)
- Python 3.8+
- PyTorch + torch_npu
- PyPTO framework
- Cannbot Skills (for Ascend C development)

## Reproducing Results

See `operators/{op}/REPRODUCE.md` for step-by-step guides for each operator.

## License

Internal research project.
