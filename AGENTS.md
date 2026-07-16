# Cannbot: Ascend C vs PyPTO — Project Agent Configuration

## Project Goal

This project compares **three implementation approaches** for each operator on Ascend NPU:

1. **Torch / torch_npu** — standard PyTorch baseline
2. **Ascend C** — kernel developed via Cannbot Skills (`ascendc-kernel-develop-workflow`)
3. **PyPTO** — kernel generated via `pypto-op-orchestrator`

Each operator goes through a unified gated pipeline: environment preflight → spec → data → torch correctness → Ascend C build/correctness → PyPTO orchestrator → unified profiler → parse → final report → dashboard → archive.

## Architecture

```
User Request
    │
    ▼
┌────────────────────────────────────────────────┐
│           Main Agent (you)                     │
│  · Scenario identification                     │
│  · Pipeline orchestration                      │
│  · Artifact gate verification                  │
│  · Unified profiler / report / archive          │
│  · Global state (AGENTS.md state NOT required) │
└──────┬──────────┬────────────────┬────────────┘
       │          │                │
       ▼          ▼                ▼
  Torch Route  Ascend C Route  PyPTO Route
  (inline)     (Cannbot Skills) (pypto-op-orchestrator)
```

---

## Route 1: Torch / torch_npu Baseline

**Owner**: Main Agent (inline, no sub-agent required)

### Steps

| Step | Artifact | Verification |
|------|----------|-------------|
| 1. Torch correctness | `torch/correctness.py` + `torch/correctness_results.json` | All batches PASS; bitwise or spec-defined tolerance |
| 2. Torch benchmark | `torch/benchmark.py` + `torch/benchmark_results.json` | Uses msprof or torch.npu.Event (documented); warmup≥200 |
| 3. Kernel mapping | Documented in final report | Profiler confirms kernel count, type, names |

### Rules
- Use the **same data** as Ascend C and PyPTO (from `data/`)
- Map each logical call to **all device kernel events** (e.g., 3x torch.add = 3 kernels)
- Do not mix JIT/Python overhead into device-kernel timing
- If torch uses fused kernels, document the kernel count difference

---

## Route 2: Ascend C (via Cannbot Skills)

**Owner**: Main Agent (orchestrates Cannbot Skills directly)

### Available Skills

The following Cannbot Skills (from `~/.config/opencode/skills/`) are used for Ascend C development. There is **no single Ascend C orchestrator** — the main agent calls skills in sequence:

| Stage | Skill | Artifact |
|-------|-------|----------|
| Spec Analysis | `ascendc-kernel-develop-workflow` | Analysis notes |
| API / Capability | `ascendc-docs-search` + `ascendc-api-best-practices` | API mapping |
| Tiling Design | `ascendc-tiling-design` | Tiling parameters |
| Kernel Design | `ascendc-kernel-develop-workflow` (stage 2) | `{op}_kernel.asc` |
| Host/Runner | `ascendc-direct-invoke-template` | `{op}_host.asc`, `CMakeLists.txt` |
| Build | `ascendc-kernel-develop-workflow` (stage 3) | Binary in `build/` |
| Correctness | `ascendc-kernel-develop-workflow` (stage 4) | Output BINs + `data/generation_scripts/correctness.py` |
| Profiler | `ascendc-kernel-develop-workflow` (stage 4) | msprof raw |
| Performance | `ops-profiling` + `ascendc-kernel-develop-workflow` (stage 5) | Parsed profiler |
| Tuning | `ascendc-precision-debug`, `ascendc-runtime-debug`, `ascendc-api-best-practices` | Optimized kernel |

### Structure

```
operators/{op}/ascendc/
├── src/
│   ├── {op}_kernel.asc        # Kernel implementation
│   ├── {op}_host.asc           # Host-side runner with <<<>>>
│   ├── {op}_tiling.h           # Tiling parameters
│   └── data_utils.h            # Shared data utilities
├── CMakeLists.txt              # Build configuration
├── build/
│   ├── {op}_ascendc            # Compiled binary
│   └── output/                 # Output BINs per batch
├── artifact_manifest.json      # Source and build metadata
└── scripts/                    # Helper scripts
```

### Rules
- Kernel must use native Ascend C API, no PyPTO
- All batches must pass correctness before entering profiler stage
- Build artifacts (.o, CMakeCache) are not tracked in SHA256SUMS
- Output BINs are stable and tracked

---

## Route 3: PyPTO (via pypto-op-orchestrator)

**Owner**: `pypto-op-orchestrator` (see `/mnt/workspace/AGENTS.md`)

### 7-Stage State Machine

| Stage | Name | Skill / Subagent | Artifact |
|-------|------|-----------------|----------|
| 1 | Intent Understanding | `pypto-intent-understand` | `SPEC.md` |
| 2 | API Exploration | `pypto-api-explore` | `API_REPORT.md` |
| 3 | Golden Generation | `pypto-op-analyst` | `{op}_golden.py` |
| 4 | Design | `pypto-op-analyst` | `DESIGN.md` |
| 5 | Implementation | `pypto-op-developer` | `{op}_impl.py`, `test_{op}.py`, `README.md` |
| 6 | Precision Fix | `pypto-op-developer` | Corrected impl (if Stage 5 fails) |
| 7 | Performance Tuning | `pypto-op-perf-tuner` | Tuned kernel (if applicable) |

### States

| State | Meaning |
|-------|---------|
| `SUCCESS` | All 7 stages completed |
| `BLOCKED_SPEC` | Stage 1 retry exhausted |
| `BLOCKED_API` | Stage 2 retry exhausted |
| `BLOCKED_GOLDEN` | Stage 3 retry exhausted |
| `BLOCKED_DESIGN` | Stage 4 retry exhausted |
| `BLOCKED_IMPL` | Stage 5 retry exhausted |
| `BLOCKED_ACCURACY` | Stage 6 retry exhausted |
| `BLOCKED_ENVIRONMENT` | Environment issue blocks all stages |

### Backend Limitation Handling
When PyPTO fails at backend (e.g., broadcast Div fails at CompileFunction):
- Mark as `BLOCKED_BACKEND` in operator status
- Document: which backend stage, error message, minimal working case
- Produce `DIAGNOSTIC_REPORT.md` with reproduction steps
- PyPTO does **not** enter the three-way performance ranking for that operator

---

## Route 4: Unified Gating (all operators)

Before any operator is marked complete, all gates must pass:

```
[1] Environment Preflight ──────────────────────────────► environment/preflight.sh
[2] SPEC & experiment_config ─────────────────────────────► SPEC.yaml, experiment_config.yaml
[3] Data Generation ─────────────────────────────────────► data/generation_scripts/
[4] Torch Correctness ───────────────────────────────────► torch/correctness_results.json (all batches PASS)
[5] Ascend C Build ──────────────────────────────────────► ascendc/build/{op}_ascendc exists
[6] Ascend C Correctness ────────────────────────────────► All batches PASS via data/generation_scripts/correctness.py
[7] PyPTO Orchestrator ──────────────────────────────────► pypto/.orchestrator_state.json (Stage 7 completed or BLOCKED)
[8] Unified Profiler ────────────────────────────────────► reports/raw/, reports/parsed/ for all 3 implementations
[9] Final Report ───────────────────────────────────────► reports/final/final_comparison.{md,json,csv}
[10] README + REPRODUCE ─────────────────────────────────► README.md, REPRODUCE.md
[11] SHA256SUMS ─────────────────────────────────────────► Source, config, final reports, parsed data (NOT build artifacts)
[12] Dashboard ──────────────────────────────────────────► dashboard/dashboard.json reflects current state
[13] Archive ────────────────────────────────────────────► tar.gz without build cache, raw profiler (optional), __pycache__
```

Any implementation that fails correctness **must not** enter the three-way performance ranking.

---

## Unified Measurement Standard

| Parameter | Default Value |
|-----------|---------------|
| Warmup | 200 iterations |
| Profiled loops | ≥ 100 iterations |
| Repeat | 5 |
| Profiler | msprof with `--ascendcl=on --ai-core=on --task-time=l0` |
| Primary metric | `primary_compute_kernel_us` (device kernel duration) |
| Secondary metric | `all_device_kernels_us_per_call` (includes executor/AICPU) |
| JIT handling | Two-process method: warmup (no profiler) → msprof session |
| Host latency | `host_synchronized_operation_us` (reported separately) |
| Kernel mapping | kernel names, types, count per logical call |

### Measurement Hierarchy

1. **Primary compute kernel** — the AI Core kernel that does the actual math
2. **All device kernels** — primary + executor/AICPU (sum, may include overlap)
3. **Host-synchronized operation** — end-to-end from Python (includes dispatch)

Do **not** rank implementations using different measurement levels.

---

## Unified Status Definitions

| Status | Definition |
|--------|------------|
| `COMPLETE` | All gates pass; full profiler data; all docs present |
| `COMPLETE_WITH_LIMITATION` | Complete but with documented backend limitation (e.g., PyPTO Div) |
| `INCOMPLETE` | Missing artifacts or failed gates |
| `BLOCKED_FRONTEND` | PyPTO frontend/JIT compilation fails |
| `BLOCKED_BACKEND` | PyPTO backend/CodeGen/CompileFunction fails |
| `NOT_COMPARABLE` | Measurement methodology differs (e.g., no msprof data) |
| `REPORT_OUTDATED` | Report claims don't match raw/parsed profiler data |
| `ARCHIVE_OUTDATED` | Archive doesn't match current source/reports |

---

## Prohibited Practices

- Hand-writing Ascend C kernel and calling it PyPTO
- Hardcoding performance results without profiler evidence
- Ranking implementations using different timing levels
- Trusting old reports without cross-verifying against raw/parsed data
- Modifying archived historical results
- Confusing API availability with backend support
- Expanding tolerance to force correctness pass
- Including build cache, object files, or CMake temp in SHA256SUMS

---

## Archive Policy

### Include in formal archive
- Source files (`ascendc/src/`, `pypto/src/`, `pypto/golden/`, `pypto/tests/`)
- Stable configs (`SPEC.yaml`, `experiment_config.yaml`, `manifest.json`)
- Correctness results (`reports/correctness/`, `torch/correctness_results.json`)
- Parsed profiler data (`reports/parsed/`)
- Final reports (`reports/final/`)
- Reproduction scripts (`REPRODUCE.md`, `benchmark/run_all.sh`)
- `SHA256SUMS` (source + stable artifacts only)
- `artifact_manifest.json`
- `.orchestrator_state.json`

### Exclude from formal archive
- Virtual environments, `__pycache__/`
- Build cache (`CMakeCache.txt`, `.o`, `CMakeFiles/`)
- Raw profiler timeline data (keep `LOCAL_ONLY`)
- JIT cache
- Full framework repositories
- Large raw data (>20MB per file) — document instead

### Path convention
All SHA256SUMS paths are relative to the operator directory (e.g., `ascendc/src/div_kernel.asc`), not project root.

---

## Operator Status Summary

| Operator | Torch | Ascend C | PyPTO | Correctness | Profiler | Report | Archive | Dashboard |
|----------|-------|----------|-------|-------------|----------|--------|---------|-----------|
| relu | COMPLETE | COMPLETE | COMPLETE | PASS (all B) | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| mul | COMPLETE | COMPLETE | COMPLETE | PASS (all B) | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| add | COMPLETE | COMPLETE | COMPLETE | PASS (all B) | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| div | COMPLETE | COMPLETE | COMPLETE_WITH_LIMITATION | PASS (all B, no PyPTO) | COMPLETE | COMPLETE | COMPLETE | COMPLETE |

### Known Limitations
- **Div PyPTO**: Broadcast Div [B,12,256,256]/[B,12,256,1] fails at backend CompileFunction. Minimal 2D Div [1,32]/[1,1] passes. PyPTO is excluded from three-way performance ranking for Div.

---

## Adding a New Operator

1. Copy template: `cp -r templates/operator_template/ operators/{op}/`
2. Fill `SPEC.yaml`, `experiment_config.yaml`
3. Generate data: `python3 data/generation_scripts/generate_inputs.py && python3 data/generation_scripts/generate_reference.py`
4. Implement torch baseline: `torch/correctness.py`, `torch/benchmark.py`
5. Develop Ascend C: use Cannbot Skills (`ascendc-kernel-develop-workflow`)
6. Generate PyPTO: use `pypto-op-orchestrator` (see `/mnt/workspace/AGENTS.md`)
7. Run unified profiler: `bash benchmark/run_all.sh`
8. Parse and report: `python3 benchmark/parse_profiler.py`
9. Generate final report: update `reports/final/final_comparison.md`
10. Verify SHA256SUMS
11. Update dashboard
12. Generate archive
