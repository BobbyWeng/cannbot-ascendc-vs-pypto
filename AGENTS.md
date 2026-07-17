# Cannbot: Ascend C vs PyPTO — Project Agent Configuration v2

## 1. Project Mission

Compare **three implementation approaches** per operator on Ascend NPU:
1. **Torch / torch_npu** — baseline
2. **Ascend C** — via `ops-direct-invoke` plugin
3. **PyPTO** — via `pypto-op-orchestrator` plugin

This project develops, repairs, optimizes, measures, and compares operators.
All Ascend C and PyPTO kernel work MUST be performed through the designated
Cannbot plugin/agent/skill workflow. Plugins drive development; they do not
remove the need for iterative implementation, debugging, and optimization.

## 2. Non-Negotiable Rules

1. **MUST classify every task** before any work — see §3.
2. **MUST load the correct plugin** — never call skills in isolation without plugin context.
3. **MUST pass G8 (Correctness)** before performance ranking — NEVER overridable.
4. **MUST pass G13 (Code Review)** before release.
5. **MUST record all state** in `reports/runtime/task_context.json` and operator `TASK_STATE.json`.
6. **MUST verify Cannbot usage** via `tools/verify_cannbot_usage.py` after any modification.
7. **MUST run `scripts/pre_release_gate.sh`** before any release.
8. **MUST run `scripts/pre_kernel_commit_gate.sh`** before any kernel commit.
9. **NO kernel logic modification** without a task context.
10. **NO fake SKILL_TRACE** — every skill invocation must be real.
11. **NO fake orchestrator state** — every stage transition must be based on real artifacts.
12. **NO report update without re-running** — stale data cannot update reports.

## 3. Task Classification

BEFORE ANY WORK: Create `reports/runtime/task_context.json` with ALL four classification axes:

### A. Backend Route
- `TORCH` — PyTorch baseline only
- `ASCENDC_DIRECT` — `<<<>>>` kernel direct invoke
- `ASCENDC_REGISTRY` — ACLNN/GEIR registry invoke
- `PYPTO` — PyPTO orchestrator
- `CATLASS` — Catlass template assembly (only when explicitly requested)
- `PROJECT_INFRA` — Project infrastructure (gates, tools, docs, audit)

### B. Semantic Class + Hardware Path

Record BOTH. They are independent axes.
| Semantic Class | Common Hardware Path | Example |
|---------------|---------------------|---------|
| ELEMENTWISE | VECTOR | relu, add, mul |
| REDUCTION | VECTOR | reduce_sum |
| MATMUL | CUBE | matmul |
| GEMM | CUBE_OR_MIXED_EPILOGUE | gemm, linear |
| LAYOUT | VECTOR_OR_SPECIALIZED_DATA_MOVE | transpose, expand |
| LOGICAL | VECTOR | equal, not, where |
| UNKNOWN_NEEDS_ANALYSIS | N/A | needs spec analysis |

### C. Task Mode
- `NEW_DEVELOPMENT` — fresh operator from scratch
- `CONTINUE_DEVELOPMENT` — resume existing development
- `RECOVER_INTERRUPTED` — recover from state file
- `FUNCTIONAL_REPAIR` — fix wrong output, crash
- `PRECISION_REPAIR` — fix precision mismatch
- `BUILD_REPAIR` — fix compilation error
- `RUNTIME_REPAIR` — fix runtime error/crash
- `PERFORMANCE_OPTIMIZATION` — improve performance
- `MEASUREMENT_AUDIT` — audit measurement methodology
- `CODE_REVIEW` — review code only
- `RELEASE_AUDIT` — audit release artifacts
- `MIGRATION` — migrate between backends
- `FRAMEWORK_PATCH` — work around backend limitation

### D. Lifecycle Stage
`PREFLIGHT` → `SPEC` → `API_ANALYSIS` → `GOLDEN_DATA` → `DESIGN` → `IMPLEMENTATION` → `BUILD` → `CORRECTNESS` → `PRECISION_FIX` → `BASELINE_PROFILE` → `OPTIMIZATION` → `FINAL_PROFILE` → `REVIEW` → `RELEASE` → `ARCHIVE`

### task_context.json Template

```json
{
  "operator": "relu",
  "backend_route": "ASCENDC_DIRECT",
  "semantic_class": "ELEMENTWISE",
  "hardware_path": "VECTOR",
  "task_mode": "NEW_DEVELOPMENT",
  "lifecycle_stage": "PREFLIGHT",
  "active_plugin": "ops-direct-invoke",
  "active_agent": "ascendc-kernel-architect",
  "required_skills": ["ascendc-env-check", "ascendc-kernel-develop-workflow", "..."],
  "loaded_skills": [],
  "cannbot_commit": "1449e954022f4733a19b243af30056901b23dd1c",
  "correctness_gate": "PENDING",
  "profile_gate": "PENDING",
  "next_action": "Run preflight.sh",
  "last_successful_checkpoint": ""
}
```

## 4. Plugin Routing

### Decision Tree

```
Task received
  │
  ├─ Backend=T0RCH → inline (no plugin)
  │
  ├─ Backend=ASCENDC_DIRECT → plugins/ops-direct-invoke
  │   ├─ Hardware=VECTOR → standard workflow + vector audit
  │   ├─ Hardware=CUBE → standard workflow + cube audit
  │   └─ Hardware=MIXED → standard workflow + both audits
  │
  ├─ Backend=ASCENDC_REGISTRY → plugins/ops-registry-invoke
  │
  ├─ Backend=PYPTO → plugins/pypto-op-orchestrator
  │
  ├─ Backend=CATLASS → plugins/catlass-op-generator
  │
  └─ Backend=PROJECT_INFRA → inline (create docs, config, tools)
```

### Plugin Responsibilities

| Plugin | Entry Point | Stages | Agents |
|--------|------------|--------|--------|
| `ops-direct-invoke` | `AGENTS.md` Step 1-7 | Env → Design → Dev → Review → Perf → Complete | architect, design-reviewer, developer, reviewer |
| `pypto-op-orchestrator` | `AGENTS.md` Stage 1-7 | Intent → API → Golden → Design → Impl → Fix → Tune | analyst, developer, perf-tuner |
| `ops-registry-invoke` | `AGENTS.md` workflow | Env → Design → Dev → Test → Review → Submit | architect, developer, tester, reviewer |
| `ops-code-reviewer` | `AGENTS.md` | Code review only | summarizer, reviewer |
| `catlass-op-generator` | `AGENTS.md` | Design → Generate → Review | architect, generator, reviewer |

## 5. Lifecycle State Machine

```
PREFLIGHT ──[G0]──> SPEC ──[G3]──> API_ANALYSIS ──[G4]──> GOLDEN_DATA ──[G5]──> DESIGN
  │                                                                                 │
  │                                                                           [G6] ▼
  │                                                                         IMPLEMENTATION
  │                                                                                 │
  │                                                                           [G7] ▼
  │                                                                          BUILD/JIT
  │                                                                                 │
  │                                                                           [G8] ▼
  │                                                                        CORRECTNESS
  │                                                                          │      │
  │                                                                          │  [G9]▼(FAIL)
  │                                                                          │ PRECISION_FIX
  │                                                                          │      │
  │                                                                          │ [G8] ▼ (re-verify)
  │                                                                          │ CORRECTNESS
  │                                                                          │      │
  │                                                                     [G10]▼      │
  │                                                               BASELINE_PROFILE   │
  │                                                                     │            │
  │                                                                [G11]▼            │
  │                                                            OPTIMIZATION          │
  │                                                                     │            │
  │                                                                [G12]▼            │
  │                                                            FINAL_PROFILE         │
  │                                                                     │            │
  │                                                                [G13]▼            │
  │                                                              CODE REVIEW         │
  │                                                                     │            │
  │                                                                [G14]▼            │
  │                                                                  RELEASE          │
  │                                                                     │            │
  │                                                                [G15]▼            │
  │                                                                  ARCHIVE          │
  │                                                                                  │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

## 6. Gate Summary

| ID | Name | Check Method | Exit |
|----|------|-------------|------|
| G0 | Environment | `environment/preflight.sh` | BLOCKED_ENVIRONMENT |
| G1 | Classification | `task_context.json` exists with 4 axes | HALT |
| G2 | Plugin/Skill Loaded | `verify_cannbot_usage.py --stage check` | HALT |
| G3 | Spec | SPEC.yaml exists, complete | RETRY |
| G4 | API Evidence | API_REPORT.md or equivalent | RETRY |
| G5 | Golden/Data | Golden runs, data exists | RETRY |
| G6 | Design Review | DESIGN.md complete, reviewed | RETRY |
| G7 | Build/JIT | Build exit 0 / no JIT error | RETRY/BLOCKED |
| G8 | Correctness | All batches PASS within tolerance | NEVER OVERRIDABLE |
| G9 | Precision | Precision meets spec tolerance | PRECISION_FIX |
| G10 | Baseline Profile | msprof raw, kernel map | RETRY |
| G11 | Optimization | ≥1 candidate evaluated | RETRY |
| G12 | Final Profile | Final msprof vs baseline | RETRY |
| G13 | Code Review | REVIEW.md PASS/PASS_WITH_NOTES | HALT |
| G14 | Release | `pre_release_gate.sh` exit 0 | HALT |
| G15 | Archive | tar.gz, SHA256SUMS verified | WARN |

Run all gates: `python3 tools/check_gate.py`
Single gate: `python3 tools/check_gate.py --gate G3`

## 7. Vector Rules

For VECTOR hardware_path:
- Must audit: GM↔UB pipeline, queue/buffer, vector API, mask/repeat, alignment, tail, blockDim, totalElements, batch stride, double buffer, scalar fallback, effective bandwidth
- Forbidden: large-scale GetValue/SetValue scalar loop, host precompute, identity kernel, prefix-only validation
- Required skills: acsendc-api-best-practices, ascendc-direct-invoke-template, ascendc-tiling-design

## 8. Cube Rules

For CUBE hardware_path:
- Must audit: M/N/K, matrix count, A/B/C layout, ND/NZ format, L1, L0A/L0B/L0C, Cube/MMAD, Fixpipe, accumulation dtype, tile M/N/K, blockDim, matrix/core, double buffer, pipeline depth, format conversion, workspace, epilogue, TFLOPS, Cube kernel evidence
- Forbidden: vector elementwise mul as MatMul, ACLNN wrapper as custom Cube, host MatMul, kernel latency without TFLOPS, hardcoded tile without shape boundary
- Required skills: ascendc-tiling-design, ascendc-api-best-practices, ascendc-blaze-best-practice, ops-profiling

## 9. Ascend C Rules

### Development (NEW_DEVELOPMENT)
1. Load `ops-direct-invoke` plugin (not scattered skills).
2. Follow Step 1-7 workflow:
   - Step 1: `ascendc-env-check` → env report
   - Step 2: `ascendc-kernel-architect` → DESIGN.md, PLAN.md
   - Step 2.5: `ascendc-kernel-design-reviewer` → WALKTHROUGH.md
   - Step 3: `ascendc-kernel-developer` → kernel code
   - Step 4: `ascendc-kernel-reviewer` → REVIEW.md
   - Step 5: Repair loop if FAIL (max 3)
   - Step 6: Precision + performance acceptance
   - Step 7: Completion report
3. All batches must pass correctness before profiler.
4. Profiler: msprof with `--ascendcl=on --ai-core=on --task-time=l0`.

### Repair (BUILD_REPAIR, RUNTIME_REPAIR)
1. Load task context → identify failure point.
2. Load skills: ascendc-runtime-debug, ascendc-env-check, ascendc-docs-search.
3. If crash: load ascending-crash-debug.
4. Fix via ascendc-kernel-developer (not inline).
5. Re-run correctness (G8) after fix.
6. Re-run profiler if optimization was involved.

### Performance (PERFORMANCE_OPTIMIZATION)
1. Verify G8 (correctness) is PASS — NO EXCEPTIONS.
2. Capture baseline profile (ops-profiling).
3. Classify bottleneck (compute-bound / memory-bound).
4. Skill-guided hypothesis (tiling grid search, double buffer, etc.).
5. Limited candidates (≤3 per iteration).
6. Correctness gate per candidate.
7. Screening profile per candidate.
8. Winner selected → full correctness + final profile.
9. Review before release.

## 10. PyPTO Rules

### Entry
- Plugin: `pypto-op-orchestrator`
- State file: `operators/{op}/.orchestrator_state.json` (official format — do NOT create custom alternative)
- 7 stages, sequentially gated
- Stage 5 has 3-way routing: PRECISION_PASS → Stage 7, PRECISION_FAIL → Stage 6, exit≠0 → retry Stage 5

### Stage Routing

| Stage | Gate | Skill | Agent |
|-------|------|-------|-------|
| 1: Intent | SPEC.md verified | pypto-intent-understand | (direct) |
| 2: API | API_REPORT.md verified | pypto-api-explore | (direct) |
| 3: Golden | {op}_golden.py runs | pypto-golden-generate | pypto-op-analyst |
| 4: Design | DESIGN.md verified | pypto-op-design | pypto-op-analyst |
| 5: Impl | test_{op}.py 3-way | pypto-op-develop | pypto-op-developer |
| 6: Fix | Re-verify | pypto-precision-debug + pypto-precision-compare | pypto-op-developer |
| 7: Perf | Profile | pypto-op-perf-tune | pypto-op-perf-tuner |

### Backend Limitation Handling
When PyPTO fails at backend:
1. Do NOT immediately mark `BLOCKED_BACKEND`.
2. Run Stage 2 API exploration again with minimal shape.
3. Check official samples.
4. Test tile/layout/dtype boundaries.
5. Try ≥3 evidence-based candidates.
6. Only then: document as BLOCKED_BACKEND with reproduction steps.

## 11. Correctness Contract

MUST cover:
- All formal batches
- Random data
- Structured data
- Zeros/ones
- Signed zero (if applicable)
- NaN propagation
- Inf behavior
- Boundary values
- Sentinel values
- Batch-unique patterns

Elementwise: bitwise when spec requires it.
Reduction/MatMul: separate accumulation dtype, FP16 ref, FP32 ref, tolerance source, error distribution.

Forbidden:
- Skipped cases counted as "all_pass"
- Missing reference but claiming PASS
- Prefix-only check
- Route-specific tolerance relaxation
- Hand-written JSON replacing real execution

## 12. Measurement Contract

| Parameter | Default |
|-----------|---------|
| Warmup | 200 iterations |
| Profiled loops | ≥ 100 |
| Repeat | 5 |
| Profiler | msprof `--ascendcl=on --ai-core=on --task-time=l0` |
| Primary metric | `primary_compute_kernel_us` |
| Secondary | `all_device_kernels_us_per_call` |
| JIT | Two-process (warmup → msprof) |
| Host latency | `host_synchronized_operation_us` (separate) |

Vector: bandwidth + bytes.
Cube: MACs, FLOPs, TFLOPS, format conversion, Cube utilization.

Forbidden:
- Event vs msprof cross-route ranking
- Per-event average as logical call
- Single tile/row as complete op
- B=1 result as full batch
- Blocked route speedup calculation
- Old binary profile in new report

## 13. NPU Scheduling

Unified queue: `reports/runtime/npu_run_queue.json`

```json
{
  "task_id": "...",
  "operator": "relu",
  "route": "ASCENDC_DIRECT",
  "stage": "BASELINE_PROFILE",
  "source_hash": "abc123",
  "command": "bash benchmark/run.sh",
  "estimated_duration_s": 300,
  "retry_count": 0,
  "status": "queued"
}
```

Rules:
- NO parallel NPU runs (queue only, one at a time).
- NO parallel PyPTO JIT.
- NO parallel Ascend C runtime.
- NO parallel benchmark/msprof/candidate timing.
- read-only tasks (doc, search, parse) CAN run in parallel.

## 14. Blocker Policy

### States
- `UNDER_INVESTIGATION` — initial state when failure is first observed.
- `BLOCKED_FRONTEND` — frontend/JIT compilation failure confirmed.
- `BLOCKED_BACKEND` — backend/CodeGen/CompileFunction failure confirmed.
- `BLOCKED_ENVIRONMENT` — environment issue (missing CANN, no NPU, etc.).
- `COMPLETE_WITH_LIMITATION` — documented limitation, not a blocker.

### Backend Blocker Requirements
Single failure is NOT sufficient for `BLOCKED_BACKEND`. Must:
1. Complete Skill/API exploration.
2. Compare with official samples.
3. Create minimal repro (max success / min fail).
4. Try ≥3 evidence-based candidates.
5. Verify frontend IR is correct.
6. Identify failing pass.
7. Check version matrix (CANN, PyPTO, framework).
8. Test environment variables.
9. Confirm stable reproduction.
10. Evaluate workarounds.
11. Evaluate framework patch.
12. Log and hash all attempts.

Failing any of these: stay `UNDER_INVESTIGATION`.

## 15. Release Policy

Single source of truth: `reports/release/current_release.json`

Chain:
```
current source/artifact → correctness → raw profiler → parsed → final comparison → current_release → dashboard → README
```

Rules:
- No report update without re-running correctness and profiler.
- No stale data in current_release.
- Historical audits are NOT inputs to dashboard.
- `scripts/pre_release_gate.sh` must exit 0 before release.

## 16. Recovery

On interruption:
1. Read `TASK_STATE.json` from operator directory.
2. Read `reports/runtime/task_context.json`.
3. Read `.orchestrator_state.json` (for PyPTO).
4. Determine last successful checkpoint.
5. Resume from the gate that was not yet passed.
6. Verify upstream artifacts are unchanged (check SHA256SUMS).
7. If upstream modified: re-verify from that gate.

## 17. Required Artifacts

### Project Level
- `reports/runtime/task_context.json` — current task classification
- `reports/runtime/project_state.json` — overall project state
- `reports/runtime/npu_run_queue.json` — NPU run schedule
- `reports/runtime/permission_deferred.json` — deferred permissions
- `reports/release/current_release.json` — release single source of truth

### Operator Level
- `TASK_STATE.json` — operator-level state (schema: `templates/shared_operator_contract/TASK_STATE.schema.json`)
- `SPEC.yaml` — operator spec
- `SKILL_TRACE.md` / `SKILL_TRACE.json` — skill invocation trace
- `SHA256SUMS` — source, config, reports hashes
- `README.md` — implementation notes
- `REPRODUCE.md` — reproduction steps
- `reports/correctness/` — correctness results
- `reports/parsed/` — parsed profiler data
- `reports/final/` — final comparison
- `.orchestrator_state.json` (PyPTO only)

## 18. Command Checklist

```
# New task
mkdir -p reports/runtime
cat > reports/runtime/task_context.json << 'EOF'
{...classification...}
EOF
python3 tools/verify_cannbot_usage.py

# Load plugin
# (plugin-specific init)

# Run gates
python3 tools/check_gate.py

# Before kernel commit
bash scripts/pre_kernel_commit_gate.sh

# Before release
bash scripts/pre_release_gate.sh
```

## Reference Files

| File | Content |
|------|---------|
| `docs/cannbot_capability_inventory.md` | Full plugin/agent/skill inventory |
| `config/skill_routing.yaml` | Skill selection matrix by task classification |
| `config/gates.yaml` | Gate definitions |
| `tools/check_gate.py` | Gate verification tool |
| `tools/verify_cannbot_usage.py` | Cannbot usage compliance check |
| `templates/shared_operator_contract/TASK_STATE.schema.json` | TASK_STATE schema |
| `templates/vector_operator_template/` | Vector operator template |
| `templates/cube_operator_template/` | Cube operator template |
