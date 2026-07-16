# Archive Checklist for {{ operator_name }}

## Before Archiving, Verify:

### Core Artifacts
- [ ] SPEC.yaml complete
- [ ] experiment_config.yaml complete
- [ ] Data generation scripts working
- [ ] Reference data generated
- [ ] Input data generated

### Torch Baseline
- [ ] benchmark.py runs without errors
- [ ] correctness.py passes

### Ascend C
- [ ] src/ has all source files
- [ ] CMakeLists.txt correct
- [ ] Builds successfully
- [ ] Correctness passes for all batches
- [ ] artifact_manifest.json complete

### PyPTO
- [ ] SPEC/SPEC.md complete
- [ ] API_REPORT/API_REPORT.md complete
- [ ] DESIGN/DESIGN.md complete
- [ ] golden/relu_golden.py works
- [ ] src/relu_impl.py works
- [ ] tests/test_relu.py passes
- [ ] artifact_manifest.json complete

### Benchmark
- [ ] profiler_config/msprof_config.json correct
- [ ] run_all.sh works
- [ ] parse_profiler.py works

### Reports
- [ ] Profiler raw data collected
- [ ] Profiler parsed data generated
- [ ] final_comparison.md produced
- [ ] final_comparison.json produced
- [ ] final_comparison.csv produced

### Documentation
- [ ] README.md written
- [ ] REPRODUCE.md written
- [ ] SHA256SUMS generated

### Cleanup
- [ ] No __pycache__ directories
- [ ] No .pytest_cache
- [ ] No build intermediates (only final artifacts)
- [ ] No large temporary files
- [ ] All paths are relative
