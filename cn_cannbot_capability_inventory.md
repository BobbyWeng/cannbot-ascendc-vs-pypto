# Cannbot 能力清单（中文版）

## 安装信息

| 属性 | 值 |
|------|-----|
| 仓库路径 | `/home/developer/.cannbot/repo` |
| Cannbot Commit | `1449e954022f4733a19b243af30056901b23dd1c` — "更新ascend2cuda工作流init脚本" |
| 安装方式 | 项目级 `init.sh`（清单: `/mnt/workspace/.opencode/cannbot-manifest.json`） |
| 工具 | OpenCode |
| 索引版本 | `2` |

## 断链检查

`/mnt/workspace/.opencode/skills/` 中的所有技能都是真实文件/目录（非符号链接）。未检测到断链。

## 已安装插件

| 插件 ID | 路径 | 已安装 | 使用中 |
|---------|------|--------|--------|
| `ops-direct-invoke` | `plugins-official/ops-direct-invoke/` | 是 | 部分 |
| `ops-direct-invoke-flash` | `plugins-official/ops-direct-invoke-flash/` | 是 | 否 |
| `ops-registry-invoke` | `plugins-official/ops-registry-invoke/` | 是 | 否 |
| `pypto-op-orchestrator` | `plugins-official/pypto-op-orchestrator/` | 是 | 是 |
| `ops-code-reviewer` | `plugins-official/ops-code-reviewer/` | 是 | 否 |
| `catlass-op-generator` | `plugins-official/catlass-op-generator/` | 是 | 否 |

## 已安装 Agents

### 来自 `/mnt/workspace/.opencode/agents/`

| Agent | 来源插件 | 使用场景 |
|-------|---------|---------|
| `ascendc-kernel-architect` | ops-direct-invoke | Ascend C 设计 |
| `ascendc-kernel-design-reviewer` | ops-direct-invoke | Ascend C 设计审查 |
| `ascendc-kernel-developer` | ops-direct-invoke | Ascend C 实现 |
| `ascendc-kernel-reviewer` | ops-direct-invoke | Ascend C 代码审查 |
| `ascendc-ops-architect` | ops-registry-invoke | 注册调用（未使用） |
| `ascendc-ops-developer` | ops-registry-invoke | 注册调用（未使用） |
| `ascendc-ops-reviewer` | ops-registry-invoke | 注册调用（未使用） |
| `ascendc-ops-tester` | ops-registry-invoke | 注册调用（未使用） |
| `ascendc-code-summarizer` | ops-code-reviewer | 代码审查 |
| `ascendc-perf-analysis-expert` | ops-direct-invoke | 性能分析 |
| `ascendc-perf-impl-expert` | ops-direct-invoke | 性能实现 |
| `catlass-op-architect` | catlass-op-generator | Catlass 设计 |
| `catlass-op-generator` | catlass-op-generator | Catlass 生成 |
| `catlass-op-reviewer` | catlass-op-generator | Catlass 审查 |
| `pypto-op-analyst` | pypto-op-orchestrator | PyPTO 阶段 3-4 |
| `pypto-op-developer` | pypto-op-orchestrator | PyPTO 阶段 5-6 |
| `pypto-op-perf-tuner` | pypto-op-orchestrator | PyPTO 阶段 7 |

## 已安装技能

### 全局（`/home/developer/.config/opencode/skills/`）

22 个 Ascend C 技能：ascendc-api-best-practices、ascendc-code-review、ascendc-direct-invoke-template、ascendc-docs-gen、ascendc-docs-search、ascendc-env-check、ascendc-kernel-develop-workflow、ascendc-npu-arch、ascendc-precision-debug、ascendc-regbase-best-practice、ascendc-runtime-debug、ascendc-st-design、ascendc-task-focus、ascendc-tiling-design、ascendc-ut-develop、ascendc-whitebox-design、cann-env-setup、ops-precision-standard、ops-profiling、ops-simulator

### 项目级（`/mnt/workspace/.opencode/skills/` — 全局之外追加）

110 个技能，包括：8 个 PyPTO 技能（pypto-api-explore、pypto-golden-generate、pypto-intent-understand、pypto-op-design、pypto-op-develop、pypto-op-perf-tune、pypto-precision-compare、pypto-precision-debug）、3 个 Catlass 技能、10 个 TileLang 技能、6 个 Triton 技能、14 个 Model 技能、8 个 Torch 技能、12 个 workflow 技能等。

## 插件 → Agent → 技能路由摘要

### ops-direct-invoke（Ascend C 直调）

| 阶段 | Agent | 技能 |
|------|-------|------|
| 环境 | 主 Agent（内联） | ascendc-env-check, ascendc-kernel-develop-workflow |
| 设计 | ascendc-kernel-architect | ascendc-tiling-design, ascendc-api-best-practices, ascendc-docs-search, npu-arch |
| 设计审查 | ascendc-kernel-design-reviewer | (使用设计文档) |
| 开发 | ascendc-kernel-developer | ascendc-direct-invoke-template, ascendc-api-best-practices, ascendc-docs-search |
| 代码审查 | ascendc-kernel-reviewer | ascendc-code-review |
| 精度调试 | ascendc-kernel-developer | ascendc-precision-debug |
| 性能 | ascendc-kernel-developer | ops-profiling, ascendc-api-best-practices, ascendc-tiling-design |

### pypto-op-orchestrator（PyPTO）

| 阶段 | 名称 | 技能 | Agent |
|------|------|------|-------|
| 1 | 意图理解 | pypto-intent-understand | (直接) |
| 2 | API 探索 | pypto-api-explore | (直接) |
| 3 | Golden 生成 | pypto-golden-generate | pypto-op-analyst |
| 4 | 设计 | pypto-op-design | pypto-op-analyst |
| 5 | 实现 | pypto-op-develop | pypto-op-developer |
| 6 | 精度修复 | pypto-precision-debug, pypto-precision-compare | pypto-op-developer |
| 7 | 性能调优 | pypto-op-perf-tune | pypto-op-perf-tuner |

## 禁止误用场景

| 插件 | 不要在以下场景使用 |
|------|-------------------|
| ops-direct-invoke | 目标需要 ACLNN/GEIR/算子仓结构；或任务是 PyPTO 特定任务 |
| ops-direct-invoke-flash | 未明确要求 Flash；需要完整工作流合规 |
| ops-registry-invoke | 仅简单 `<<<>>>` 基准测试；无需框架集成 |
| pypto-op-orchestrator | 任务是 Ascend C 原生开发（改用 ops-direct-invoke） |
| catlass-op-generator | Cube 算子更适合 PyPTO Cube 或使用 Cube API 的 ops-direct-invoke |
| ops-code-reviewer | 审查非 Ascend C 代码；快速内联检查即可 |

## 版本/提交信息

| 项目 | 值 |
|------|-----|
| Cannbot 仓库提交 | `1449e95` — "更新ascend2cuda工作流init脚本" |
| 前一提交 | `f937b96` — "[doc]新增tilelang flashAttention 算子调优skill" |
| 索引版本 | `2` |
| ops-direct-invoke 版本 | `1.0.0` |
| ops-registry-invoke 版本 | `1.0.0` |
| 项目清单 | `/mnt/workspace/.opencode/cannbot-manifest.json`（安装: 2026-07-15T09:06:34Z, 团队: pypto-op-orchestrator） |

## 关键发现

1. PyPTO 技能和 agent 安装在项目级（`/mnt/workspace/.opencode/skills/` 和 `agents/`）。
2. Ascend C 技能同时安装在全局和项目级（重复）。
3. 仓库中不存在 `ops-lab/` 目录 — 所有操作都在 `ops/` 中。
4. `ops-direct-invoke` 工作流可用但**未被项目充分利用** — 旧 AGENTS.md 逐个调用技能而非使用插件的 agent。
5. `ops-code-reviewer` 插件可用但从未接入项目流水线。
