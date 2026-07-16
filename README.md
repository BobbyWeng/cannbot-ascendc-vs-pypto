# Cannbot: Ascend C vs PyPTO Operator Comparison

A structured, reproducible comparison framework for evaluating Ascend C and PyPTO operator implementations against torch baselines on Ascend NPU hardware.

## Project Structure

```
cannbot_ascendc_vs_pypto/
├── README.md                        # This file
├── AGENTS.md                        # Agent configuration
├── environment/                     # Environment and version manifests
│   ├── environment_manifest.json
│   └── preflight.sh
├── common/                          # Shared libraries
│   ├── schemas/                     # JSON schemas
│   ├── benchmark/                   # Benchmark utilities
│   ├── correctness/                 # Correctness checking
│   ├── profiler/                    # Profiler parsing
│   └── reporting/                   # Report generation
├── operators/                       # Operator comparison directories
│   ├── relu/                        # ReLU (COMPLETE)
│   ├── add/                         # Addition (COMPLETE)
│   ├── mul/                         # Multiplication (COMPLETE)
│   ├── div/                         # Division (COMPLETE, PyPTO backend limitation)
│   ├── equal/                       # Equal (COMPLETE_WITH_LIMITATION)
│   ├── not/                         # LogicalNot (REPORT_OUTDATED - needs correction)
│   ├── or/                          # LogicalOr (REPORT_OUTDATED - needs correction)
│   ├── where/                       # Where (COMPLETE_WITH_LIMITATION)
│   ├── expand/                      # Expand (INCOMPLETE)
│   ├── transpose/                   # Transpose (INCOMPLETE)
│   ├── reduce_sum/                  # ReduceSum (INCOMPLETE)
├── templates/                       # Templates for new operators
│   └── operator_template/
├── reports/                         # Project-level reports
├── scripts/                         # Project-level scripts
└── AGENTS.md
```

## Operator Directory Structure

Each operator under `operators/{op}/` follows a consistent structure:

```
operators/{op}/
├── README.md                        # Operator-specific README
├── SPEC.yaml                        # Operator specification
├── experiment_config.yaml           # Experiment configuration
├── data/                            # Input data and reference
│   ├── manifest.json
│   └── generation_scripts/
├── torch/                           # Torch baseline
│   ├── benchmark.py
│   └── correctness.py
├── ascendc/                         # Ascend C implementation
│   ├── src/
│   ├── CMakeLists.txt
│   ├── build/
│   ├── scripts/
│   └── artifact_manifest.json
├── pypto/                           # PyPTO implementation
│   ├── SPEC/
│   ├── API_REPORT/
│   ├── DESIGN/
│   ├── golden/
│   ├── src/
│   ├── tests/
│   ├── scripts/
│   └── artifact_manifest.json
├── benchmark/                       # Benchmark runners
│   ├── run_all.sh
│   ├── profiler_config/
│   └── parse_profiler.py
├── reports/                         # Results
│   ├── raw/
│   ├── parsed/
│   ├── correctness/
│   └── final/
├── REPRODUCE.md                     # Reproduction guide
└── SHA256SUMS                       # File integrity hashes
```

## Adding a New Operator

To add a new operator, use the template:

```bash
cp -r templates/operator_template/ operators/{op_name}/
# Then customize all {{ variable }} placeholders
```

Refer to the `templates/operator_template/archive_checklist.md` for the complete verification checklist.

## Operator Status

| Operator | Torch Baseline | Ascend C | PyPTO | Correctness | Profiler | Report | Status |
|----------|---------------|----------|-------|-------------|----------|--------|--------|
| relu     | ✅ COMPLETE | ✅ TRUE_DEVICE_IMPLEMENTATION | ✅ SUCCESS | ✅ PASS (all B) | ✅ msprof | ✅ Complete | ✅ COMPLETE |
| mul      | ✅ COMPLETE | ✅ TRUE_DEVICE_IMPLEMENTATION | ✅ SUCCESS | ✅ PASS (all B) | ✅ msprof | ✅ Complete | ✅ COMPLETE |
| add      | ✅ COMPLETE | ✅ TRUE_DEVICE_IMPLEMENTATION | ✅ SUCCESS | ✅ PASS (all B) | ✅ msprof | ✅ Complete | ✅ COMPLETE |
| div      | ✅ COMPLETE | ✅ TRUE_DEVICE_IMPLEMENTATION | ⚠️ BLOCKED_BACKEND | ✅ Torch+AscendC PASS; PyPTO limited | ✅ msprof (B=32) | ✅ Complete | ✅ COMPLETE_WITH_LIMITATION |
| equal    | ✅ COMPLETE | ✅ TRUE_DEVICE_IMPLEMENTATION | ❌ BLOCKED_BACKEND_EQUAL | ✅ Torch+AscendC PASS | ⚠️ torch.npu.Event (NOT_COMPARABLE) | ✅ Complete | ✅ COMPLETE_WITH_LIMITATION |
| not      | ✅ COMPLETE | ✅ TRUE_DEVICE_IMPLEMENTATION | ✅ PASS | ❌ AscendC FAIL (script bug) | ⚠️ torch.npu.Event (NOT_COMPARABLE) | ⚠️ OUTDATED | ⚠️ REPORT_OUTDATED |
| or       | ✅ COMPLETE | ✅ TRUE_DEVICE_IMPLEMENTATION | ⚠️ PARTIAL (bitwise_or bug) | ❌ AscendC FAIL (script bug) | ⚠️ torch.npu.Event (NOT_COMPARABLE) | ⚠️ OUTDATED | ⚠️ REPORT_OUTDATED |
| where    | ✅ COMPLETE | ✅ TRUE_DEVICE_WITH_SCALAR_FALLBACK | ❌ BLOCKED_BACKEND_WHERE_SELECT | ✅ Torch+AscendC PASS | ⚠️ torch.npu.Event (NOT_COMPARABLE) | ✅ Complete | ✅ COMPLETE_WITH_LIMITATION |
| expand   | ⚠️ B=1 only | ❌ HOST_PRECOMPUTE_FALLBACK | ⚠️ PARTIAL | ⬜ NOT_RUN | ⬜ N/A | ⚠️ Overstated | ❌ INCOMPLETE |
| transpose| ⚠️ B=1 only | ❌ HOST_PRECOMPUTE_FALLBACK | ❌ BLOCKED_BACKEND (large) | ⬜ NOT_RUN | ⬜ N/A | ⚠️ Overstated | ❌ INCOMPLETE |
| reduce_sum| ⚠️ B=1 only| ❌ HOST_PRECOMPUTE_FALLBACK | ✅ SUCCESS | ⬜ NOT_RUN | ⬜ N/A | ⚠️ Overstated | ❌ INCOMPLETE |

## Measurement Methodology

All operators use a **unified profiler-based measurement** approach:

1. **Profiler**: msprof with `--ascendcl=on --ai-core=on --task-time=l0`
2. **Warmup**: 200 iterations (excluded from measurement)
3. **Profiled iterations**: 100 iterations
4. **Key metric**: `all_device_kernels_us` — sum of all device kernel durations per logical call
5. **PyPTO**: Two-process method (warmup no-profiler, then msprof) to exclude JIT compilation time

## Requirements

- Ascend 910B or compatible NPU
- CANN Toolkit (tested with 9.0.0)
- Python 3.8+
- PyTorch + torch_npu
- PyPTO framework
- Cannbot Skills (for Ascend C development)
- pypto-op-orchestrator (for PyPTO development)

## Reproducing Results

See `operators/{op}/REPRODUCE.md` for step-by-step guides for each operator.

## License

Internal research project.
