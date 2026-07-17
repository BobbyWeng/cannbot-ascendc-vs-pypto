# Ascend C 工作流（中文版）

## 入口
- 插件: `ops-direct-invoke`（来自 `/mnt/workspace/.opencode/`）
- Agents: `ascendc-kernel-architect`, `ascendc-kernel-design-reviewer`, `ascendc-kernel-developer`, `ascendc-kernel-reviewer`
- 技能: 见 `config/skill_routing.yaml` 和 `docs/SKILL_SELECTION_MATRIX.md`

## 7 步工作流

```
Step 1: 环境检查 → ascendc-env-check → environment.md
Step 2: 设计 → ascendc-kernel-architect → DESIGN.md + PLAN.md
Step 2.5: 设计审查 → ascendc-kernel-design-reviewer → WALKTHROUGH.md
Step 3: 开发 → ascendc-kernel-developer → 内核代码, 构建
Step 4: 代码审查 → ascendc-kernel-reviewer → REVIEW.md
  ├─ PASS / PASS WITH NOTES → Step 6
  └─ FAIL → Step 5（最多 3 轮修复）
Step 5: 修复循环（最多 3 轮）
Step 6: 精度 + 性能验收（Reviewer 精度验证 + Developer 性能采集）
Step 7: 完成报告
```

## 路线特定指令

### Vector（hardware_path=VECTOR）
- 使用 Vector API（DataCopy, Add, Sub 等）
- 设计必须包含：GM↔UB 流水线、队列/缓冲区、mask/repeat、对齐、尾部、blockDim、totalElements、有效带宽
- 技能：ascendc-api-best-practices、ascendc-tiling-design、ascendc-direct-invoke-template

### Cube（hardware_path=CUBE）
- 使用 Cube API（Mmad, Matmul, Fixpipe, LoadData）
- 设计必须包含：M/N/K、布局、L1/L0A/L0B/L0C、MMAD、TFLOPS
- 技能：ascendc-blaze-best-practice、ascendc-tiling-design、ascendc-performance-best-practices

### 性能优化
- G8 正确性必须在任何优化前通过
- 通过 ops-profiling（msprof）基线 profile
- 瓶颈分类（计算密集 vs 内存密集）
- 每轮有限候选项（≤3）
- 每个候选项通过正确性门禁
- 胜出者需要完整正确性 + 最终 profile

### 构建/运行时修复
- 加载 ascendc-runtime-debug
- 若崩溃：加载 ascendc-crash-debug
- 通过 developer agent 修复（不由主 agent 内联）
- 修复后重新运行正确性
- 若涉及优化则重新运行 profiler

### 精度修复
- 加载 ascendc-precision-debug
- 对照精度标准（ops-precision-standard）比较
- 不要放宽容差以强制 PASS
- 通过 ascendc-kernel-reviewer 审查

## 与项目集成

```
operators/{op}/
├── ascendc/
│   ├── src/         # 内核源文件
│   ├── CMakeLists.txt
│   ├── build/       # 构建输出
│   └── scripts/
├── reports/
│   ├── correctness/  # Ascend C 正确性
│   └── parsed/       # Ascend C profiler 数据
└── ...
```
