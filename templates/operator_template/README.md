# {{ operator_name }} Operator — torch / Ascend C / PyPTO Comparison

This directory contains a full comparison of {{ operator_name }} across three implementation approaches on Ascend NPU.

## Contents
- `SPEC.yaml` — Operator specification
- `experiment_config.yaml` — Experiment configuration
- `data/` — Input data and reference files
- `torch/` — Torch baseline (benchmark + correctness)
- `ascendc/` — Ascend C implementation
- `pypto/` — PyPTO implementation (via pypto-op-orchestrator)
- `benchmark/` — Profiler-based benchmark runner
- `reports/` — Raw, parsed, correctness, and final reports
- `REPRODUCE.md` — Step-by-step reproduction guide
- `SHA256SUMS` — File integrity hashes
