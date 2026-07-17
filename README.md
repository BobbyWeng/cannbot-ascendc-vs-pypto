# Cannbot: Ascend C vs PyPTO Operator Comparison — v1.1 (RC-1)

A structured, reproducible comparison framework for evaluating Ascend C and PyPTO operator implementations against torch baselines on Ascend NPU hardware.

## Project Structure

```
cannbot_ascendc_vs_pypto/
├── README.md                        # This file
├── AGENTS.md                        # Agent configuration
├── environment/                     # Environment and version manifests
├── common/                          # Shared libraries (profiler, correctness, benchmark)
├── operators/                       # 12 Operator comparison directories
│   ├── relu/                        # ReLU (COMPLETE)
│   ├── mul/                         # Multiplication (COMPLETE)
│   ├── add/                         # Addition (COMPLETE_WITH_LIMITATION)
│   ├── div/                         # Division (COMPLETE_WITH_LIMITATION)
│   ├── equal/                       # Equal (COMPLETE_WITH_LIMITATION)
│   ├── not/                         # LogicalNot (COMPLETE)
│   ├── or/                          # LogicalOr (COMPLETE_WITH_LIMITATION)
│   ├── where/                       # Where (COMPLETE_WITH_LIMITATION)
│   ├── expand/                      # Expand (COMPLETE_WITH_LIMITATION)
│   ├── transpose/                   # Transpose (COMPLETE_WITH_LIMITATION)
│   ├── reduce_sum/                  # ReduceSum (COMPLETE_WITH_LIMITATION)
│   └── matmul/                      # MatMul (COMPLETE)
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
│   ├── validation_freeze/           # Validation Freeze audit reports
│   └── operator_summary.md/json     # Quick-reference summary
├── dashboard/                       # Standalone dashboard
├── scripts/                         # NPU lock, profiler queue scripts
└── archives/                        # Operator archive packages
```

## Operator Status (RC-1)

| Operator | Final Status | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------------|-------|----------|-------|-------------|----------|
| relu | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full (7/7) | msprof |
| mul | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full (7/7) | msprof |
| not | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full (42/42) | Event |
| matmul | **COMPLETE** | PASS | TRUE_CUBE | BLOCKED_BACKEND | Torch+AscendC (6/6) | msprof |
| add | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | PyPTO B=1 persisted | msprof |
| div | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BLOCKED_BACKEND | Torch+AscendC (6/6) | msprof |
| equal | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BLOCKED_BACKEND | Torch+AscendC (7/7) | Event |
| or | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BITWISE_OR | AscendC corrected (49/49) | Event |
| where | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BLOCKED_BACKEND | Torch+AscendC (7/7) | Event |
| expand | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | Full (7/7) | msprof(r3) |
| transpose | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | PARTIAL | Torch+AscendC (7/7) | msprof |
| reduce_sum | **COMPLETE_WITH_LIMITATION** | PASS (62/70) | TRUE_DEVICE | SUCCESS (21/70) | FP16 accum | msprof |

**Status**: 4 COMPLETE, 8 COMPLETE_WITH_LIMITATION

## RC-1 Fixes

- **Profiler parser**: `primary_compute_kernel_us` now correctly reports KERNEL_MIX_AIC for PyPTO (not KERNEL_AICPU executor)
- **Div correctness**: B=4,8,16,32 reference files existed, re-ran — all 6 batches PASS now
- **Relu correctness**: torch/correctness_results.json was missing, now generated — 7/7 PASS
- **Reduce_sum parsed data**: 14 parsed JSON files generated from existing msprof raw data
- **Performance CSV**: All values use consistent `primary_compute_kernel_us` metric

## Measurement Methodology

All msprof operators use **unified profiler-based measurement**:
1. **Profiler**: msprof with `--ascendcl=on --ai-core=on --task-time=l0`
2. **Warmup**: 200 iterations (excluded from measurement)
3. **Profiled iterations**: 100 iterations
4. **Key metric**: `primary_compute_kernel_us` — longest single device kernel event per call
5. **PyPTO**: Two-process method (warmup no-profiler, then msprof) to exclude JIT compilation

**Event-based operators** (equal, not, or, where): Use `torch.npu.Event` / `aclrtEvent` host-synchronized timing. NOT comparable with msprof operators.

## Known Limitations

- **PyPTO backend limitations**: div (broadcast), equal (output bitmask), matmul (Cube tiling FC4000), transpose (CompileFunction >1K elements), where (condition dtype expansion)
- **No msprof for logical ops**: equal, not, or, where lack device-kernel profiling
- **Reduce_sum FP16 accumulation**: Ascend C and PyPTO at 21/70 PASS due to FP16 vs FP32 reference
- **Expand per-row dispatch**: PyPTO uses 256 JIT invocations per batch row (AICPU, not compute kernel)
- **Matmul per-matrix dispatch**: Ascend C launches 12-384 individual kernel calls

## Dashboard

```bash
python dashboard/dashboard.py --release reports/release/current_release.json
```

Open `dashboard/index.html` in any browser.

## Requirements

- Ascend 910B or compatible NPU
- CANN Toolkit (tested with 9.0.0)
- Python 3.8+
- PyTorch + torch_npu
- PyPTO framework
- Cannbot Skills (for Ascend C development)

## Reproducing Results

See `reports/validation_freeze/` for complete audit reports.

See `operators/{op}/REPRODUCE.md` for step-by-step guides for each operator.
