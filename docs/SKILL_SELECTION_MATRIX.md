# Skill Selection Matrix

This document defines which skills to load based on task classification.
The canonical routing rules are in `config/skill_routing.yaml`. This is the human-readable reference.

## Match Logic

For each task, determine:
1. `backend` — from task_context.json `backend_route`
2. `task_mode` — from task_context.json `task_mode`
3. `hardware_path` — from task_context.json `hardware_path`

Then find the matching routing rule in `config/skill_routing.yaml`.

## Quick Reference

### Ascend C — New Development

| Stage | Skills | Agent |
|-------|--------|-------|
| Preflight | ascendc-env-check | (main agent) |
| Design | ascendc-kernel-develop-workflow, ascendc-tiling-design, ascendc-api-best-practices, ascendc-docs-search | ascendc-kernel-architect |
| Design Review | (uses design docs) | ascendc-kernel-design-reviewer |
| Implementation | ascendc-direct-invoke-template, ascendc-api-best-practices, ascendc-docs-search | ascendc-kernel-developer |
| Review | ascendc-code-review | ascendc-kernel-reviewer |
| Precision | ascendc-precision-debug, ops-precision-standard | ascendc-kernel-developer |
| Performance | ops-profiling, ascendc-tiling-design, ascendc-api-best-practices | ascendc-perf-analysis-expert, ascendc-kernel-developer |

**Vector add**: ascendc-api-best-practices (vector-specific)
**Cube add**: ascendc-blaze-best-practice, ascendc-performance-best-practices

### Ascend C — Build/Runtime Repair

| Stage | Skills | Agent |
|-------|--------|-------|
| Diagnose | ascendc-runtime-debug, ascendc-env-check, ascendc-docs-search | ascendc-kernel-developer |
| Fix | ascendc-kernel-develop-workflow | ascendc-kernel-developer |
| Review | ascendc-code-review | ascendc-kernel-reviewer |
| *If crash* | ascendc-crash-debug | (add to diagnose) |

### Ascend C — Precision Repair

| Stage | Skills | Agent |
|-------|--------|-------|
| Diagnose | ascendc-precision-debug, ops-precision-standard, ascendc-docs-search | ascendc-kernel-developer |
| Fix | ascendc-api-best-practices | ascendc-kernel-developer |
| Review | ascendc-code-review | ascendc-kernel-reviewer |

**Forbidden**: Expending tolerance to force PASS.

### Ascend C — Performance Optimization

| Stage | Skills | Agent |
|-------|--------|-------|
| Baseline | ops-profiling, ascendc-npu-arch | ascendc-perf-analysis-expert |
| Optimize | ascendc-tiling-design, ascendc-api-best-practices, ascendc-perf-optimize | ascendc-kernel-developer, ascendc-perf-impl-expert |
| Review | ascendc-code-review | ascendc-kernel-reviewer |
| *Cube add* | ascendc-blaze-best-practice | (add to optimize) |

**Forbidden**: Parameter sweep without profiler evidence.

### PyPTO — New Development

| Stage | Skills | Agent |
|-------|--------|-------|
| 1: Intent | pypto-intent-understand | (direct) |
| 2: API | pypto-api-explore | (direct) |
| 3: Golden | pypto-golden-generate | pypto-op-analyst |
| 4: Design | pypto-op-design | pypto-op-analyst |
| 5: Impl | pypto-op-develop | pypto-op-developer |
| 6: Fix | pypto-precision-debug, pypto-precision-compare | pypto-op-developer |
| 7: Perf | pypto-op-perf-tune | pypto-op-perf-tuner |

### PyPTO — Build/Backend Repair

| Stage | Skills | Agent |
|-------|--------|-------|
| Recover | (task_context + .orchestrator_state.json) | (main agent) |
| API Re-explore | pypto-api-explore | (direct) |
| Minimal Repro | pypto-op-design, pypto-op-develop | pypto-op-developer |
| Fix | pypto-op-developer | pypto-op-developer |
| Verify | pypto-op-develop | pypto-op-developer |
| *If precision* | pypto-precision-debug | (add) |
| *If slow* | pypto-op-perf-tune | pypto-op-perf-tuner |

### PyPTO — Performance Optimization

| Stage | Skills | Agent |
|-------|--------|-------|
| Baseline | pypto-op-perf-tune | pypto-op-perf-tuner |
| Tune | (pypto-op-perf-tune sub-skills: frontend/incore/swimlane) | pypto-op-perf-tuner |

**Prerequisite**: G8 (Correctness) must be PASS.

## Architecture-specific Requirements

### VECTOR
- ascendc-api-best-practices (vector APIs: DataCopy, Add, Sub, Mul, etc.)
- ascendc-tiling-design (UB split, double buffer)
- Audit: GM↔UB pipeline, queue/buffer, mask/repeat, alignment, tail, blockDim

### CUBE
- ascendc-blaze-best-practice (MatMul/Cube best practices)
- ascendc-tiling-design (M/N/K tiling)
- ascendc-performance-best-practices (Cube performance reference)
- Audit: M/N/K layout, L1/L0A/L0B/L0C, MMAD, Fixpipe, TFLOPS
