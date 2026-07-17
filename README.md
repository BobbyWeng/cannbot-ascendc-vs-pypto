# Cannbot: Ascend C vs PyPTO Operator Comparison — v1.3-rc3

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
│   ├── rc3/                         # RC-3 final reports
│   └── operator_summary.md/json     # Quick-reference summary
├── dashboard/                       # Standalone dashboard
├── scripts/                         # NPU lock, profiler queue scripts
└── archives/                        # Operator archive packages
```

## Operator Status (v1.3-rc3)

| Operator | Final Status | Torch | Ascend C | PyPTO | Correctness | Profiler |
|----------|-------------|-------|----------|-------|-------------|----------|
| relu | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full (7/7) | msprof |
| mul | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full (7/7) | msprof |
| not | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | Full (42/42) | msprof |
| matmul | **COMPLETE** | PASS | TRUE_CUBE | SUCCESS (limited) | All 3 routes (6/6) | msprof |
| add | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | PyPTO B=1 persisted | msprof |
| div | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | All 3 routes (6/6 bitwise) | msprof |
| equal | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | All 3 routes (7/7 bitwise) | msprof |
| or | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BITWISE_OR | AscendC corrected (49/49) | msprof |
| where | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | All 3 routes (7/7 bitwise) | msprof |
| expand | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | Full (7/7) | msprof |
| transpose | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | All 3 routes (7/7 bitwise) | msprof |
| reduce_sum | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | 70/70 (FP32 accum) | msprof |

**Status**: 4 COMPLETE, 8 COMPLETE_WITH_LIMITATION, 0 PARTIAL, 0 INCOMPLETE

## Measurement Methodology

All msprof operators use **unified profiler-based measurement**:
1. **Profiler**: msprof with `--ascendcl=on --ai-core=on --task-time=l0`
2. **Warmup**: 200 iterations (excluded from measurement)
3. **Profiled iterations**: 100 iterations
4. **Key metric**: `primary_compute_kernel_us` — longest single device kernel event per call
5. **PyPTO**: Two-process method (warmup no-profiler, then msprof) to exclude JIT compilation

## Known Limitations

| Priority | Operator | Route | Description |
|----------|----------|-------|-------------|
| P1 | or | PyPTO | Uses bitwise_or (no logical_or API). Correct for 0/1 bool. |
| P1 | reduce_sum | all | FP16 output overflow >65504 (expected behavior) |
| P2 | matmul | PyPTO | Auto-tiling FC4000; manual set_cube_tile_shapes required |
| P2 | equal | PyPTO | BOOL output requires ta≤64 tile constraint |
| P2 | where | PyPTO | uint8 condition requires DT_BOOL kernel |
| P2 | expand | PyPTO | Uses PyTorch expand+clone (not PyPTO native) |
| P2 | add | PyPTO | Correctness B=2..64 not persisted to JSON |

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

See `reports/rc3/` for RC-3 final audit reports. Historical reports are in `archives/rc_history/`.

See `operators/{op}/REPRODUCE.md` for step-by-step guides for each operator.
