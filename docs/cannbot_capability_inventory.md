# Cannbot Capability Inventory

## Installation Facts

| Property | Value |
|----------|-------|
| Repo Path | `/home/developer/.cannbot/repo` |
| Cannbot Commit | `1449e954022f4733a19b243af30056901b23dd1c` — "更新ascend2cuda工作流init脚本" |
| Install Mode | Project-level via `init.sh` (manifest at `/mnt/workspace/.opencode/cannbot-manifest.json`) |
| Tool | OpenCode |
| Index Version | `2` |

## Broken Symlinks Check

All skills at `/mnt/workspace/.opencode/skills/` are real files/directories (not symlinks). No broken links detected.

## Installed Plugins

| Plugin ID | Path | Installed | In Use |
|-----------|------|-----------|--------|
| `ops-direct-invoke` | `plugins-official/ops-direct-invoke/` | Yes (available) | Partial |
| `ops-direct-invoke-flash` | `plugins-official/ops-direct-invoke-flash/` | Yes (available) | No |
| `ops-registry-invoke` | `plugins-official/ops-registry-invoke/` | Yes (available) | No |
| `pypto-op-orchestrator` | `plugins-official/pypto-op-orchestrator/` | Yes (installed via skill) | Yes |
| `ops-code-reviewer` | `plugins-official/ops-code-reviewer/` | Yes (available) | No |
| `catlass-op-generator` | `plugins-official/catlass-op-generator/` | Yes (available) | No |

## Installed Agents

### From `/mnt/workspace/.opencode/agents/`

| Agent | Source Plugin | Used In |
|-------|--------------|---------|
| `ascendc-kernel-architect` | ops-direct-invoke | Ascend C Development |
| `ascendc-kernel-design-reviewer` | ops-direct-invoke | Ascend C Design Review |
| `ascendc-kernel-developer` | ops-direct-invoke | Ascend C Implementation |
| `ascendc-kernel-reviewer` | ops-direct-invoke | Ascend C Code Review |
| `ascendc-ops-architect` | ops-registry-invoke | Registry Invoke (not used) |
| `ascendc-ops-developer` | ops-registry-invoke | Registry Invoke (not used) |
| `ascendc-ops-reviewer` | ops-registry-invoke | Registry Invoke (not used) |
| `ascendc-ops-tester` | ops-registry-invoke | Registry Invoke (not used) |
| `ascendc-code-summarizer` | ops-code-reviewer | Code Review |
| `ascendc-perf-analysis-expert` | ops-direct-invoke | Performance Analysis |
| `ascendc-perf-impl-expert` | ops-direct-invoke | Performance Implementation |
| `catlass-op-architect` | catlass-op-generator | Catlass Design |
| `catlass-op-generator` | catlass-op-generator | Catlass Generation |
| `catlass-op-reviewer` | catlass-op-generator | Catlass Review |
| `pypto-op-analyst` | pypto-op-orchestrator | PyPTO Stages 3-4 |
| `pypto-op-developer` | pypto-op-orchestrator | PyPTO Stages 5-6 |
| `pypto-op-perf-tuner` | pypto-op-orchestrator | PyPTO Stage 7 |

### From `/home/developer/.cannbot/repo/plugins-official/` (available but not project-installed)

| Agent | Source Plugin |
|-------|--------------|
| `model-infer-analyzer` | model-infer-optimize |
| `model-infer-implementer` | model-infer-optimize |
| `model-infer-reviewer` | model-infer-optimize |
| `model-infer-sota-*` (6 agents) | model-infer-sota-approach |
| `tilelang-op-analyst` | tilelang-op-orchestrator |
| `tilelang-op-developer` | tilelang-op-orchestrator |
| `tilelang-op-perf-tuner` | tilelang-op-orchestrator |
| `tilelang2ascendc-kernel-generator` | tilelang2ascendc-ops-generator |

## Installed Skills

### Global (at `/home/developer/.config/opencode/skills/`)

| Skill | Category | Loaded |
|-------|----------|--------|
| ascendc-api-best-practices | Knowledge | Yes |
| ascendc-code-review | Test/Review | Yes |
| ascendc-custom-op-to-kernel-launch | Migration | Yes |
| ascendc-direct-invoke-template | Template | Yes |
| ascendc-docs-gen | Documentation | Yes |
| ascendc-docs-search | Knowledge | Yes |
| ascendc-env-check | Tool | Yes |
| ascendc-kernel-develop-workflow | Workflow | Yes |
| ascendc-npu-arch | Knowledge | Yes |
| ascendc-precision-debug | Debug/Diagnosis | Yes |
| ascendc-regbase-best-practice | Knowledge | Yes |
| ascendc-registry-invoke-to-direct-invoke | Migration | Yes |
| ascendc-runtime-debug | Debug/Diagnosis | Yes |
| ascendc-st-design | Test | Yes |
| ascendc-task-focus | Tool | Yes |
| ascendc-tiling-design | Knowledge | Yes |
| ascendc-ut-develop | Test | Yes |
| ascendc-whitebox-design | Test | Yes |
| cann-env-setup | Tool | Yes |
| ops-precision-standard | Knowledge | Yes |
| ops-profiling | Performance | Yes |
| ops-simulator | Simulation | Yes |

### Project-level (at `/mnt/workspace/.opencode/skills/` — additional beyond global)

| Skill | Category | Loaded |
|-------|----------|--------|
| aiss-tiling-solver | Knowledge | Yes |
| ascendc-blaze-best-practice | Knowledge | Yes |
| ascendc-crash-debug | Debug | Yes |
| ascendc-direct-invoke-to-registry-invoke | Migration | Yes |
| ascendc-mc2-best-practice | Knowledge | Yes |
| ascendc-perf-optimize | Performance | Yes |
| ascendc-performance-best-practices | Knowledge | Yes |
| ascendc-simt-best-practices | Knowledge | Yes |
| ascendc-simt-tiling-design | Knowledge | Yes |
| ascendc-registry-invoke-template | Template | Yes |
| catlass-op-design | Knowledge | Yes |
| catlass-op-develop | Template | Yes |
| catlass-op-perf-tune | Performance | Yes |
| model-infer-* (14 skills) | Model | Yes |
| model-train-* (3 skills) | Model | Yes |
| npu-arch | Knowledge | Yes |
| ops-direct-invoke-flash | Workflow | Yes |
| ops-direct-invoke-workflow | Workflow | Yes |
| ops-direct-invoke-workflow-maintain | Workflow | Yes |
| ops-precision-standard | Knowledge | Yes |
| ops-profiling | Performance | Yes |
| ops-simulator | Simulation | Yes |
| ops-spec-gen | Documentation | Yes |
| **pypto-api-explore** | **API** | **Yes** |
| **pypto-golden-generate** | **Golden** | **Yes** |
| **pypto-intent-understand** | **Spec** | **Yes** |
| **pypto-op-design** | **Design** | **Yes** |
| **pypto-op-develop** | **Develop** | **Yes** |
| **pypto-op-perf-tune** | **Perf** | **Yes** |
| **pypto-precision-compare** | **Debug** | **Yes** |
| **pypto-precision-debug** | **Debug** | **Yes** |
| repo-* (5 skills) | Repository | Yes |
| runtime_migration | Runtime | Yes |
| tilelang-* (10 skills) | TileLang | Yes |
| tilelang2ascend-* (6 skills) | TileLang | Yes |
| torch-* (8 skills) | Torch | Yes |
| triton-* (6 skills) | Triton | Yes |
| workflow-* (12 skills) | Workflows | Yes |

## Plugin → Agent → Skill Routing Summary

### ops-direct-invoke (Ascend C Direct Invoke)

| Stage | Agent | Skills |
|-------|-------|--------|
| Environment | Main Agent (inline) | ascendc-env-check, ascendc-kernel-develop-workflow |
| Design | ascendc-kernel-architect | ascendc-tiling-design, ascendc-api-best-practices, ascendc-docs-search, npu-arch |
| Design Review | ascendc-kernel-design-reviewer | (uses design docs as input) |
| Development | ascendc-kernel-developer | ascendc-direct-invoke-template, ascendc-api-best-practices, ascendc-docs-search |
| Code Review | ascendc-kernel-reviewer | ascendc-code-review |
| Precision Debug | ascendc-kernel-developer | ascendc-precision-debug |
| Performance | ascendc-kernel-developer | ops-profiling, ascendc-api-best-practices, ascendc-tiling-design |

### ops-direct-invoke-flash (Flash variant)

Simpler pipeline: uses `ops-direct-invoke-flash` skill directly. No separate agent stages.

### ops-registry-invoke (Registry Invoke)

| Stage | Agent | Skills |
|-------|-------|--------|
| Design | ascendc-ops-architect | ops-registry-invoke-workflow |
| Development | ascendc-ops-developer | ops-registry-invoke-workflow |
| Test | ascendc-ops-tester | ascendc-st-design, ascendc-whitebox-design |
| Review | ascendc-ops-reviewer | ascendc-code-review |

### pypto-op-orchestrator (PyPTO)

| Stage | Name | Skill | Agent |
|-------|------|-------|-------|
| 1 | Intent Understanding | pypto-intent-understand | (direct) |
| 2 | API Exploration | pypto-api-explore | (direct) |
| 3 | Golden Generation | pypto-golden-generate | pypto-op-analyst |
| 4 | Design | pypto-op-design | pypto-op-analyst |
| 5 | Implementation | pypto-op-develop | pypto-op-developer |
| 6 | Precision Fix | pypto-precision-debug, pypto-precision-compare | pypto-op-developer |
| 7 | Performance Tune | pypto-op-perf-tune | pypto-op-perf-tuner |

### catlass-op-generator (Catlass)

| Stage | Agent | Skills |
|-------|-------|--------|
| Design | catlass-op-architect | catlass-op-design |
| Generation | catlass-op-generator | catlass-op-develop |
| Review | catlass-op-reviewer | catlass-op-perf-tune |

### ops-code-reviewer (Code Review)

Single stage: load `ascendc-code-review` skill and route to its workflow.
Sub-agents: `ascendc-code-summarizer`, `ascendc-ops-reviewer`.

## Prohibited Misuse Scenarios

| Plugin | Do NOT use when |
|--------|-----------------|
| ops-direct-invoke | Target requires ACLNN/GEIR/算子仓 structure; task is PyPTO-specific |
| ops-direct-invoke-flash | Flash not specifically requested; full workflow compliance needed |
| ops-registry-invoke | Simple direct benchmark with `<<<>>>` only; no framework integration needed |
| pypto-op-orchestrator | Task is Ascend C native development (use ops-direct-invoke instead) |
| catlass-op-generator | CUBE operator is better served by PyPTO Cube or ops-direct-invoke with Cube APIs |
| ops-code-reviewer | Reviewing non-Ascend-C code; quick inline check suffices |

## Version/Commit Information

| Item | Value |
|------|-------|
| Cannbot Repo Commit | `1449e95` — "更新ascend2cuda工作流init脚本，与直调工作流init脚本入参对齐" |
| Cannbot Repo Previous | `f937b96` — "[doc]新增tilelang flashAttention 算子调优skill" |
| Index Version | `2` |
| ops-direct-invoke Version | `1.0.0` |
| ops-registry-invoke Version | `1.0.0` |
| Project Manifest | `/mnt/workspace/.opencode/cannbot-manifest.json` (Installed: 2026-07-15T09:06:34Z, Team: pypto-op-orchestrator) |

## Key Findings

1. PyPTO skills and agents are installed at project level (`/mnt/workspace/.opencode/skills/` and `agents/`).
2. Ascend C skills are installed both globally and at project level (duplicated).
3. No `ops-lab/` directory exists in the repo — all ops are in `ops/`.
4. The `ops-direct-invoke` workflow is available but was **not being fully utilized** by the project — the old AGENTS.md called skills ad-hoc instead of using the plugin's agents.
5. `ops-direct-invoke-flash` is available but should only be used when the flash template variant is explicitly needed.
6. `catlass-op-generator` agents and skills are installed but unused in current project.
7. The `ops-code-reviewer` plugin is available but was never wired into the project's pipeline.
