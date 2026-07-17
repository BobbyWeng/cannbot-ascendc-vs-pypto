# 技能选择矩阵（中文版）

本文档定义了基于任务分类加载哪些技能。规范路由规则在 `config/skill_routing.yaml` 中。此文件为人类可读参考。

## 匹配逻辑

对于每个任务，确定：
1. `backend` — 来自 task_context.json 的 `backend_route`
2. `task_mode` — 来自 task_context.json 的 `task_mode`
3. `hardware_path` — 来自 task_context.json 的 `hardware_path`

然后在 `config/skill_routing.yaml` 中找到匹配的路由规则。

## 快速参考

### Ascend C — 新开发

| 阶段 | 技能 | Agent |
|------|------|-------|
| 前置检查 | ascendc-env-check | (主 agent) |
| 设计 | ascendc-kernel-develop-workflow, ascendc-tiling-design, ascendc-api-best-practices, ascendc-docs-search | ascendc-kernel-architect |
| 设计审查 | (使用设计文档) | ascendc-kernel-design-reviewer |
| 实现 | ascendc-direct-invoke-template, ascendc-api-best-practices, ascendc-docs-search | ascendc-kernel-developer |
| 审查 | ascendc-code-review | ascendc-kernel-reviewer |
| 精度 | ascendc-precision-debug, ops-precision-standard | ascendc-kernel-developer |
| 性能 | ops-profiling, ascendc-tiling-design, ascendc-api-best-practices | ascendc-perf-analysis-expert, ascendc-kernel-developer |

**Vector 追加**: ascendc-api-best-practices（vector 特定）
**Cube 追加**: ascendc-blaze-best-practice, ascendc-performance-best-practices

### Ascend C — 构建/运行时修复

| 阶段 | 技能 | Agent |
|------|------|-------|
| 诊断 | ascendc-runtime-debug, ascendc-env-check, ascendc-docs-search | ascendc-kernel-developer |
| 修复 | ascendc-kernel-develop-workflow | ascendc-kernel-developer |
| 审查 | ascendc-code-review | ascendc-kernel-reviewer |
| *若崩溃* | ascendc-crash-debug | (追加到诊断) |

### Ascend C — 精度修复

| 阶段 | 技能 | Agent |
|------|------|-------|
| 诊断 | ascendc-precision-debug, ops-precision-standard, ascendc-docs-search | ascendc-kernel-developer |
| 修复 | ascendc-api-best-practices | ascendc-kernel-developer |
| 审查 | ascendc-code-review | ascendc-kernel-reviewer |

**禁止**: 放宽容差以强制 PASS。

### Ascend C — 性能优化

| 阶段 | 技能 | Agent |
|------|------|-------|
| 基线 | ops-profiling, ascendc-npu-arch | ascendc-perf-analysis-expert |
| 优化 | ascendc-tiling-design, ascendc-api-best-practices, ascendc-perf-optimize | ascendc-kernel-developer, ascendc-perf-impl-expert |
| 审查 | ascendc-code-review | ascendc-kernel-reviewer |
| *Cube 追加* | ascendc-blaze-best-practice | (追加到优化) |

**禁止**: 无 profiler 证据的参数扫描。

### PyPTO — 新开发

| 阶段 | 技能 | Agent |
|------|------|-------|
| 1: 意图 | pypto-intent-understand | (直接) |
| 2: API | pypto-api-explore | (直接) |
| 3: Golden | pypto-golden-generate | pypto-op-analyst |
| 4: 设计 | pypto-op-design | pypto-op-analyst |
| 5: 实现 | pypto-op-develop | pypto-op-developer |
| 6: 修复 | pypto-precision-debug, pypto-precision-compare | pypto-op-developer |
| 7: 性能 | pypto-op-perf-tune | pypto-op-perf-tuner |

### PyPTO — 构建/后端修复

| 阶段 | 技能 | Agent |
|------|------|-------|
| 恢复 | (task_context + .orchestrator_state.json) | (主 agent) |
| API 重新探索 | pypto-api-explore | (直接) |
| 最小复现 | pypto-op-design, pypto-op-develop | pypto-op-developer |
| 修复 | pypto-op-developer | pypto-op-developer |
| 验证 | pypto-op-develop | pypto-op-developer |
| *若精度* | pypto-precision-debug | (追加) |
| *若慢* | pypto-op-perf-tune | pypto-op-perf-tuner |

### PyPTO — 性能优化

| 阶段 | 技能 | Agent |
|------|------|-------|
| 基线 | pypto-op-perf-tune | pypto-op-perf-tuner |
| 调优 | (pypto-op-perf-tune 子技能: frontend/incore/swimlane) | pypto-op-perf-tuner |

**前置条件**: G8（正确性）必须为 PASS。

## 架构特定要求

### VECTOR
- ascendc-api-best-practices（vector API: DataCopy, Add, Sub, Mul 等）
- ascendc-tiling-design（UB 切分、双缓冲）
- 审计: GM↔UB 流水线、队列/缓冲区、mask/repeat、对齐、尾部、blockDim

### CUBE
- ascendc-blaze-best-practice（MatMul/Cube 最佳实践）
- ascendc-tiling-design（M/N/K tiling）
- ascendc-performance-best-practices（Cube 性能参考）
- 审计: M/N/K 布局、L1/L0A/L0B/L0C、MMAD、Fixpipe、TFLOPS
