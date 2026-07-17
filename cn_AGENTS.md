# Cannbot: Ascend C vs PyPTO — 项目 Agent 配置 v2（中文版）

## 1. 项目使命

在昇腾 NPU 上对**三种实现路线**进行逐算子对比：
1. **Torch / torch_npu** — 基线
2. **Ascend C** — 通过 `ops-direct-invoke` 插件
3. **PyPTO** — 通过 `pypto-op-orchestrator` 插件

本项目**不是**算子开发项目——它是一个**治理、测量和对比框架**。此处不开发任何内核逻辑；所有实现均由插件生成。

## 2. 不可协商规则

1. **任何工作前必须先分类**任务 — 见 §3。
2. **必须加载正确的插件** — 不得脱离插件上下文单独调用技能。
3. **必须通过 G8（正确性）** 之后才能进入性能排名 — **绝不可覆盖**。
4. **必须通过 G13（代码审查）** 之后才能发布。
5. **必须将所有状态记录**在 `reports/runtime/task_context.json` 和算子 `TASK_STATE.json` 中。
6. **任何修改后必须验证 Cannbot 使用**，通过 `tools/verify_cannbot_usage.py`。
7. **发布前必须运行** `scripts/pre_release_gate.sh`。
8. **提交内核代码前必须运行** `scripts/pre_kernel_commit_gate.sh`。
9. **没有任务上下文时禁止修改内核逻辑**。
10. **禁止伪造 SKILL_TRACE** — 每次技能调用必须是真实的。
11. **禁止伪造 orchestrator 状态** — 每次阶段转换必须基于真实工件。
12. **禁止未重新运行就更新报告** — 过时数据不能更新报告。

## 3. 任务分类

在开始任何工作前，创建 `reports/runtime/task_context.json`，包含以下全部四个分类轴：

### A. 后端路线
- `TORCH` — 仅 PyTorch 基线
- `ASCENDC_DIRECT` — `<<<>>>` 内核直接调用
- `ASCENDC_REGISTRY` — ACLNN/GEIR 注册调用
- `PYPTO` — PyPTO 编排器
- `CATLASS` — Catlass 模板组装（仅在明确要求时）
- `PROJECT_INFRA` — 项目基础设施（门禁、工具、文档、审计）

### B. 语义类别 + 硬件路径

同时记录两者。它们是独立的轴。
| 语义类别 | 常见硬件路径 | 示例 |
|---------|-------------|------|
| ELEMENTWISE | VECTOR | relu, add, mul |
| REDUCTION | VECTOR | reduce_sum |
| MATMUL | CUBE | matmul |
| GEMM | CUBE_OR_MIXED_EPILOGUE | gemm, linear |
| LAYOUT | VECTOR_OR_SPECIALIZED_DATA_MOVE | transpose, expand |
| LOGICAL | VECTOR | equal, not, where |
| UNKNOWN_NEEDS_ANALYSIS | 无 | 需要规格分析 |

### C. 任务模式
- `NEW_DEVELOPMENT` — 全新算子开发
- `CONTINUE_DEVELOPMENT` — 继续已有开发
- `RECOVER_INTERRUPTED` — 从状态文件恢复
- `FUNCTIONAL_REPAIR` — 修复输出错误、崩溃
- `PRECISION_REPAIR` — 修复精度不匹配
- `BUILD_REPAIR` — 修复编译错误
- `RUNTIME_REPAIR` — 修复运行时错误/崩溃
- `PERFORMANCE_OPTIMIZATION` — 提升性能
- `MEASUREMENT_AUDIT` — 审计测量方法
- `CODE_REVIEW` — 仅代码审查
- `RELEASE_AUDIT` — 审计发布工件
- `MIGRATION` — 后端间迁移
- `FRAMEWORK_PATCH` — 绕过后端限制

### D. 生命周期阶段
`PREFLIGHT` → `SPEC` → `API_ANALYSIS` → `GOLDEN_DATA` → `DESIGN` → `IMPLEMENTATION` → `BUILD` → `CORRECTNESS` → `PRECISION_FIX` → `BASELINE_PROFILE` → `OPTIMIZATION` → `FINAL_PROFILE` → `REVIEW` → `RELEASE` → `ARCHIVE`

### task_context.json 模板

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
  "next_action": "运行 preflight.sh",
  "last_successful_checkpoint": ""
}
```

## 4. 插件路由

### 决策树

```
收到任务
  │
  ├─ 后端=TORCH → 内联（无插件）
  │
  ├─ 后端=ASCENDC_DIRECT → plugins/ops-direct-invoke
  │   ├─ 硬件=VECTOR → 标准工作流 + vector 审计
  │   ├─ 硬件=CUBE → 标准工作流 + cube 审计
  │   └─ 硬件=MIXED → 标准工作流 + 双重审计
  │
  ├─ 后端=ASCENDC_REGISTRY → plugins/ops-registry-invoke
  │
  ├─ 后端=PYPTO → plugins/pypto-op-orchestrator
  │
  ├─ 后端=CATLASS → plugins/catlass-op-generator
  │
  └─ 后端=PROJECT_INFRA → 内联（创建文档、配置、工具）
```

### 插件职责

| 插件 | 入口点 | 阶段 | Agents |
|------|--------|------|--------|
| `ops-direct-invoke` | `AGENTS.md` Step 1-7 | 环境→设计→开发→审查→性能→完成 | architect, design-reviewer, developer, reviewer |
| `pypto-op-orchestrator` | `AGENTS.md` Stage 1-7 | 意图→API→Golden→设计→实现→修复→调优 | analyst, developer, perf-tuner |
| `ops-registry-invoke` | `AGENTS.md` 工作流 | 环境→设计→开发→测试→审查→提交 | architect, developer, tester, reviewer |
| `ops-code-reviewer` | `AGENTS.md` | 仅代码审查 | summarizer, reviewer |
| `catlass-op-generator` | `AGENTS.md` | 设计→生成→审查 | architect, generator, reviewer |

## 5. 生命周期状态机

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
  │                                                                          │  [G9]▼(失败)
  │                                                                          │ PRECISION_FIX
  │                                                                          │      │
  │                                                                          │ [G8] ▼ (重新验证)
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

## 6. 门禁摘要

| ID | 名称 | 检查方法 | 退出状态 |
|----|------|---------|---------|
| G0 | 环境 | `environment/preflight.sh` | BLOCKED_ENVIRONMENT |
| G1 | 分类 | `task_context.json` 存在且含 4 轴 | HALT |
| G2 | 插件/技能加载 | `verify_cannbot_usage.py --stage check` | HALT |
| G3 | 规格 | SPEC.yaml 存在且完整 | RETRY |
| G4 | API 证据 | API_REPORT.md 或等效文件 | RETRY |
| G5 | Golden/数据 | Golden 可运行，数据存在 | RETRY |
| G6 | 设计审查 | DESIGN.md 完整且已审查 | RETRY |
| G7 | 构建/JIT | 构建退出 0 / 无 JIT 错误 | RETRY/BLOCKED |
| G8 | 正确性 | 所有批次在容差范围内通过 | **绝不可覆盖** |
| G9 | 精度 | 精度满足规格容差 | PRECISION_FIX |
| G10 | 基线 Profile | msprof 原始数据，内核映射 | RETRY |
| G11 | 优化 | ≥1 个候选项已评估 | RETRY |
| G12 | 最终 Profile | 最终 msprof 对比基线 | RETRY |
| G13 | 代码审查 | REVIEW.md PASS/PASS_WITH_NOTES | HALT |
| G14 | 发布 | `pre_release_gate.sh` 退出 0 | HALT |
| G15 | 归档 | tar.gz, SHA256SUMS 验证 | WARN |

运行所有门禁：`python3 tools/check_gate.py`
单个门禁：`python3 tools/check_gate.py --gate G3`

## 7. Vector 规则

对于 VECTOR hardware_path：
- 必须审计：GM↔UB 流水线、队列/缓冲区、vector API、mask/repeat、对齐、尾部处理、blockDim、totalElements、batch 跨度、双缓冲、标量回退、有效带宽
- 禁止：大规模 GetValue/SetValue 标量循环、主机预计算、恒等内核、仅验证输出前缀
- 必需技能：ascendc-api-best-practices、ascendc-direct-invoke-template、ascendc-tiling-design

## 8. Cube 规则

对于 CUBE hardware_path：
- 必须审计：M/N/K、矩阵计数、A/B/C 布局、ND/NZ 格式、L1、L0A/L0B/L0C、Cube/MMAD、Fixpipe、累积 dtype、tile M/N/K、blockDim、每核矩阵数、双缓冲、流水线深度、格式转换、workspace、epilogue、TFLOPS、Cube 内核证据
- 禁止：vector 逐元素乘冒充 MatMul、ACLNN wrapper 冒充自定义 Cube、主机 MatMul、无 TFLOPS 的内核延迟、无形状边界的硬编码 tile
- 必需技能：ascendc-tiling-design、ascendc-api-best-practices、ascendc-blaze-best-practice、ops-profiling

## 9. Ascend C 规则

### 开发（NEW_DEVELOPMENT）
1. 加载 `ops-direct-invoke` 插件（不是分散的技能）。
2. 遵循 Step 1-7 工作流：
   - Step 1: `ascendc-env-check` → 环境报告
   - Step 2: `ascendc-kernel-architect` → DESIGN.md, PLAN.md
   - Step 2.5: `ascendc-kernel-design-reviewer` → WALKTHROUGH.md
   - Step 3: `ascendc-kernel-developer` → 内核代码
   - Step 4: `ascendc-kernel-reviewer` → REVIEW.md
   - Step 5: 若 FAIL 则修复循环（最多 3 轮）
   - Step 6: 精度 + 性能验收
   - Step 7: 完成报告
3. 所有批次必须在 profiler 前通过正确性。
4. Profiler: msprof 参数 `--ascendcl=on --ai-core=on --task-time=l0`。

### 修复（BUILD_REPAIR, RUNTIME_REPAIR）
1. 加载任务上下文 → 识别失败点。
2. 加载技能：ascendc-runtime-debug、ascendc-env-check、ascendc-docs-search。
3. 若崩溃：加载 ascendc-crash-debug。
4. 通过 ascendc-kernel-developer 修复（非内联）。
5. 修复后重新运行正确性（G8）。
6. 若涉及优化则重新运行 profiler。

### 性能（PERFORMANCE_OPTIMIZATION）
1. 验证 G8（正确性）已通过 — **无例外**。
2. 捕获基线 profile（ops-profiling）。
3. 分类瓶颈（计算密集型 / 内存密集型）。
4. 技能引导的假设（tiling 网格搜索、双缓冲等）。
5. 有限候选项（每轮 ≤3）。
6. 每个候选项通过正确性门禁。
7. 每个候选项筛选 profile。
8. 选择胜出者 → 完整正确性 + 最终 profile。
9. 发布前审查。

## 10. PyPTO 规则

### 入口
- 插件: `pypto-op-orchestrator`
- 状态文件: `operators/{op}/.orchestrator_state.json`（官方格式 — 不要创建自定义替代）
- 7 阶段，顺序门禁
- Stage 5 有三态路由：PRECISION_PASS → Stage 7, PRECISION_FAIL → Stage 6, exit≠0 → 重试 Stage 5

### 阶段路由

| 阶段 | 门禁 | 技能 | Agent |
|------|------|------|-------|
| 1: 意图 | SPEC.md 已验证 | pypto-intent-understand | (直接) |
| 2: API | API_REPORT.md 已验证 | pypto-api-explore | (直接) |
| 3: Golden | {op}_golden.py 可运行 | pypto-golden-generate | pypto-op-analyst |
| 4: 设计 | DESIGN.md 已验证 | pypto-op-design | pypto-op-analyst |
| 5: 实现 | test_{op}.py 三态 | pypto-op-develop | pypto-op-developer |
| 6: 修复 | 重新验证 | pypto-precision-debug + pypto-precision-compare | pypto-op-developer |
| 7: 性能 | Profile | pypto-op-perf-tune | pypto-op-perf-tuner |

### 后端限制处理
当 PyPTO 在后端失败时：
1. 不要立即标记 `BLOCKED_BACKEND`。
2. 使用最小 shape 重新运行 Stage 2 API 探索。
3. 检查官方样例。
4. 测试 tile/layout/dtype 边界。
5. 尝试 ≥3 个基于证据的候选项。
6. 只有在之后：记录为 BLOCKED_BACKEND 并附带复现步骤。

## 11. 正确性契约

必须覆盖：
- 所有正式批次
- 随机数据
- 结构化数据
- 零/全一
- 有符号零（若适用）
- NaN 传播
- Inf 行为
- 边界值
- 哨兵值
- 批次唯一模式

逐元素：规格要求时需位精确。
归约/MatMul：分别记录累积 dtype、FP16 参考、FP32 参考、容差来源、误差分布。

禁止：
- 跳过的用例计入 "all_pass"
- 缺少参考但声称 PASS
- 仅检查前缀
- 为单条路线放宽容差
- 手写 JSON 替代真实执行

## 12. 测量契约

| 参数 | 默认值 |
|------|--------|
| 预热 | 200 次迭代 |
| 测量循环 | ≥ 100 |
| 重复 | 5 |
| Profiler | msprof `--ascendcl=on --ai-core=on --task-time=l0` |
| 主要指标 | `primary_compute_kernel_us` |
| 次要指标 | `all_device_kernels_us_per_call` |
| JIT | 双进程（预热 → msprof） |
| 主机延迟 | `host_synchronized_operation_us`（单独报告） |

Vector：带宽 + 字节数。
Cube：MACs、FLOPs、TFLOPS、格式转换、Cube 利用率。

禁止：
- Event 与 msprof 跨路线排名
- per-event 平均值冒充逻辑调用
- 单 tile/单行冒充完整操作
- B=1 结果冒充全批次
- 阻塞路线的加速比计算
- 旧二进制 profile 进入新报告

## 13. NPU 调度

统一队列：`reports/runtime/npu_run_queue.json`

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

规则：
- **禁止并行 NPU 运行**（仅队列，一次一个）。
- **禁止并行 PyPTO JIT**。
- **禁止并行 Ascend C 运行时**。
- **禁止并行 benchmark/msprof/候选项计时**。
- 只读任务（文档、搜索、解析）**可以**并行。

## 14. 阻塞策略

### 状态
- `UNDER_INVESTIGATION` — 首次观察到失败时的初始状态。
- `BLOCKED_FRONTEND` — 前端/JIT 编译失败已确认。
- `BLOCKED_BACKEND` — 后端/CodeGen/CompileFunction 失败已确认。
- `BLOCKED_ENVIRONMENT` — 环境问题（缺少 CANN、无 NPU 等）。
- `COMPLETE_WITH_LIMITATION` — 已记录限制，非阻塞。

### 后端阻塞要求
单次失败不足以标记 `BLOCKED_BACKEND`。必须：
1. 完成技能/API 探索。
2. 与官方样例对比。
3. 创建最小复现（最大成功/最小失败）。
4. 尝试 ≥3 个基于证据的候选项。
5. 验证前端 IR 正确。
6. 识别失败的 pass。
7. 检查版本矩阵（CANN、PyPTO、框架）。
8. 测试环境变量。
9. 确认稳定复现。
10. 评估 workaround。
11. 评估 framework patch。
12. 记录所有尝试的日志和哈希。

任何一项未满足：保持 `UNDER_INVESTIGATION`。

## 15. 发布策略

单一事实源：`reports/release/current_release.json`

链式：
```
当前源代码/工件 → 正确性 → 原始 profiler → 解析 → 最终对比 → current_release → 仪表盘 → README
```

规则：
- 未重新运行正确性和 profiler 之前不得更新报告。
- current_release 中不得有过时数据。
- 历史审计不是仪表盘的输入。
- `scripts/pre_release_gate.sh` 必须在发布前退出 0。

## 16. 恢复

中断时：
1. 从算子目录读取 `TASK_STATE.json`。
2. 读取 `reports/runtime/task_context.json`。
3. 读取 `.orchestrator_state.json`（PyPTO）。
4. 确定最后成功的检查点。
5. 从尚未通过的门禁恢复。
6. 验证上游工件未更改（检查 SHA256SUMS）。
7. 若上游被修改：从该门禁重新验证。

## 17. 必需工件

### 项目级别
- `reports/runtime/task_context.json` — 当前任务分类
- `reports/runtime/project_state.json` — 项目整体状态
- `reports/runtime/npu_run_queue.json` — NPU 运行调度
- `reports/runtime/permission_deferred.json` — 延迟权限
- `reports/release/current_release.json` — 发布单一事实源

### 算子级别
- `TASK_STATE.json` — 算子级状态（schema: `templates/shared_operator_contract/TASK_STATE.schema.json`）
- `SPEC.yaml` — 算子规格
- `SKILL_TRACE.md` / `SKILL_TRACE.json` — 技能调用跟踪
- `SHA256SUMS` — 源代码、配置、报告的哈希
- `README.md` — 实现说明
- `REPRODUCE.md` — 复现步骤
- `reports/correctness/` — 正确性结果
- `reports/parsed/` — 已解析的 profiler 数据
- `reports/final/` — 最终对比
- `.orchestrator_state.json`（仅 PyPTO）

## 18. 命令检查清单

```
# 新任务
mkdir -p reports/runtime
cat > reports/runtime/task_context.json << 'EOF'
{...分类...}
EOF
python3 tools/verify_cannbot_usage.py

# 加载插件
#（插件特定的初始化）

# 运行门禁
python3 tools/check_gate.py

# 提交内核前
bash scripts/pre_kernel_commit_gate.sh

# 发布前
bash scripts/pre_release_gate.sh
```

## 参考文件

| 文件 | 内容 |
|------|------|
| `docs/cannbot_capability_inventory.md` | 完整的插件/Agent/技能清单 |
| `config/skill_routing.yaml` | 按任务分类的技能选择矩阵 |
| `config/gates.yaml` | 门禁定义 |
| `tools/check_gate.py` | 门禁验证工具 |
| `tools/verify_cannbot_usage.py` | Cannbot 使用合规检查 |
| `templates/shared_operator_contract/TASK_STATE.schema.json` | TASK_STATE 模式 |
| `templates/vector_operator_template/` | Vector 算子模板 |
| `templates/cube_operator_template/` | Cube 算子模板 |
