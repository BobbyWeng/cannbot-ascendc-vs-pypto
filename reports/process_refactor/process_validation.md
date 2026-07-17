# Process Refactor: Validation Results

## Dry-Run Scenarios

### Scenario 1: Vector New Development (ReLU-like)
- **Classification**: ASCENDC_DIRECT / ELEMENTWISE / VECTOR / NEW_DEVELOPMENT
- **Plugin**: ops-direct-invoke
- **Agents**: architect → design-reviewer → developer → reviewer
- **Skills**: ascendc-env-check, ascendc-kernel-develop-workflow, ascendc-direct-invoke-template, ascendc-tiling-design, ascendc-api-best-practices, ops-precision-standard
- **Vector-specific audit**: GM↔UB pipeline, queue/buffer, mask/repeat, alignment, tail, blockDim, effective bandwidth
- **Gates**: G0→G15 (all 16)
- **G8**: NEVER overridable

### Scenario 2: Cube Performance Optimization (MatMul-like)
- **Classification**: ASCENDC_DIRECT / MATMUL / CUBE / PERFORMANCE_OPTIMIZATION
- **Plugin**: ops-direct-invoke
- **Agents**: perf-analysis-expert → developer/perf-impl-expert → reviewer
- **Skills**: ops-profiling, ascendc-tiling-design, ascendc-api-best-practices, ascendc-npu-arch, ascendc-blaze-best-practice, ascendc-perf-optimize
- **Prerequisite**: G8 (correctness) must be PASS — verified by gate
- **Forbidden**: Optimization without profiler evidence
- **Cube audit**: M/N/K, layout, L1/L0A/L0B/L0C, MMAD, Fixpipe, TFLOPS

### Scenario 3: PyPTO CompileFunction Repair
- **Classification**: PYPTO / ELEMENTWISE / VECTOR / FRAMEWORK_PATCH
- **Plugin**: pypto-op-orchestrator
- **Agents**: orchestrator (main) → analyst → developer
- **Skills**: pypto-api-explore, pypto-op-design, pypto-op-develop, pypto-precision-debug (fallback)
- **Must NOT block immediately**: systematic diagnosis with ≥3 candidates required
- **Orchestrator state**: restored from `.orchestrator_state.json`

### Scenario 4: Flow Replay on ReLU
- **Existing artifacts**: SPEC.yaml, data/, torch/correctness_results.json, ascendc/build/, pypto/.orchestrator_state.json, SKILL_TRACE files
- **Gates identified as passed**: G3, G5, G7, G8, PyPTO Stages 1-7 all completed
- **Missing (new requirement)**: TASK_STATE.json — needs creation
- **Next action**: Create TASK_STATE.json, verify SHA256SUMS unchanged, mark gates as passed

## Tool Validation

| Tool | Test | Result |
|------|------|--------|
| check_gate.py --gate G1 | No task_context.json | Correctly FAIL |
| check_gate.py --gate G1 | With valid task_context.json | Correctly PASS |
| verify_cannbot_usage.py | Full check without records | Correctly FAIL (no skills loaded, no SKILL_TRACE, no NPU queue) |
| verify_cannbot_usage.py | Has task_context + plugin | Correctly PASS those sub-checks |
| pre_kernel_commit_gate.sh | No correctness run | Correctly FAIL (G8 not passed) |
| pre_kernel_commit_gate.sh | Has task_context | Correctly PASS classification check |

## Key Findings

1. All existing operators (relu, matmul, add, div, etc.) lack TASK_STATE.json — this is the most impactful new requirement.
2. Existing operators have SKILL_TRACE files, which is good.
3. The PyPTO orchestrator states are correct and usable.
4. No NPU run queue was being used — NPU scheduling was ad-hoc.
5. No code review gate was ever enforced — G13 was bypassed.
6. The correctness gate G8 was respected conceptually but never mechanically enforced.
