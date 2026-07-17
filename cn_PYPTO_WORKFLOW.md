# PyPTO 工作流（中文版）

## 入口
- 插件: `pypto-op-orchestrator`（来自 `/mnt/workspace/.opencode/`）
- 状态: `operators/{op}/.orchestrator_state.json`（官方格式）
- Agents: `pypto-op-analyst`, `pypto-op-developer`, `pypto-op-perf-tuner`

## 7 阶段状态机

| 阶段 | 名称 | 技能 | Agent | 工件 |
|------|------|------|-------|------|
| 1 | 意图理解 | pypto-intent-understand | (直接) | SPEC.md |
| 2 | API 探索 | pypto-api-explore | (直接) | API_REPORT.md |
| 3 | Golden 生成 | pypto-golden-generate | pypto-op-analyst | {op}_golden.py |
| 4 | 设计 | pypto-op-design | pypto-op-analyst | DESIGN.md |
| 5 | 实现 | pypto-op-develop | pypto-op-developer | {op}_impl.py, test_{op}.py |
| 6 | 精度修复 | pypto-precision-debug, pypto-precision-compare | pypto-op-developer | 修正后的实现 |
| 7 | 性能调优 | pypto-op-perf-tune | pypto-op-perf-tuner | 调优后的内核 |

## Stage 5 路由

```
实现结果
    │
    ├─ [PRECISION_PASS] → Stage 7
    │
    ├─ [PRECISION_FAIL] → Stage 6
    │
    └─ exit code ≠ 0（无标记）
        ├─ 编译错误 → 重试 Stage 5（修复编译）
        ├─ ImportError → BLOCKED_ENVIRONMENT
        ├─ AiCore 错误 → 报告，评估
        ├─ shape 不匹配 → 重试 Stage 5（修复 shape）
        └─ 其他运行时 → 重试 Stage 5（修复运行时）
```

**重要**: orchestrator 必须独立重新运行精度测试以确认 agent 的声明，然后再进行路由。

## 后端限制处理

当 PyPTO 在后端失败时：
1. 不要立即阻塞。进行系统诊断：
2. 使用最小 shape 重新探索 API（Stage 2）。
3. 检查官方 PyPTO 样例。
4. 测试 shape/dtype/layout 边界。
5. 检查前端 IR。
6. 识别失败的 pass（lowering → CompileFunction）。
7. 检查版本矩阵（CANN、PyPTO 版本）。
8. 尝试 ≥3 个基于证据的候选项。
9. 若全部失败：记录为 BLOCKED_BACKEND 并附带复现步骤。

## 状态文件

使用官方的 `.orchestrator_state.json` 格式。不要创建自定义等效文件。

```json
{
  "operator_name": "{op}",
  "current_stage": 5,
  "stage_status": {"1": "completed", ...},
  "stage_retry_count": {"1": 0, ...},
  "perf_iteration": {"count": 0, ...},
  "last_updated": "2026-07-17T00:00:00Z"
}
```

## 与项目集成

```
operators/{op}/
├── pypto/
│   ├── {op}_golden.py
│   ├── {op}_impl.py
│   ├── test_{op}.py
│   └── .orchestrator_state.json
├── reports/
│   ├── correctness/    ← PyPTO 正确性结果
│   └── parsed/         ← PyPTO profiler 结果
└── ...
```
