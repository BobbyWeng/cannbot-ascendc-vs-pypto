# Cube 算子工作流（中文版）

适用于 `hardware_path: CUBE` 的算子。

示例算子：matmul, batch_matmul, linear, qk, pv, ffn, gemm。

## 工作流

与父工作流（见 `cn_OPERATOR_DEVELOPMENT_WORKFLOW.md`）相同，增加 Cube 特定内容。

### 决策树

```
请求 Cube 算子
    │
    ├─ 明确要求 Catlass → catlass-op-generator
    │   Agents: catlass-op-architect → catlass-op-generator → catlass-op-reviewer
    │
    ├─ PyPTO Cube → pypto-op-orchestrator
    │   标准 7 阶段 PyPTO 流程
    │
    └─ Ascend C 原生 Cube → ops-direct-invoke
       标准 7 步流程 + Cube 审计要求
```

### 设计（Ascend C）
Agent: `ascendc-kernel-architect`
技能: `ascendc-kernel-develop-workflow`, `ascendc-tiling-design`, `ascendc-api-best-practices`, `ascendc-blaze-best-practice`, `ascendc-docs-search`

必须记录：
- M, N, K 维度
- 每次逻辑调用的矩阵计数
- A, B, C 布局（ND/NZ）
- L1 缓冲区使用
- L0A/L0B/L0C 配置
- Cube MMAD 设置
- Fixpipe 配置
- 累积 dtype
- Tile M/N/K 大小
- blockDim 分配
- 每核矩阵数
- 双缓冲深度
- 流水线深度
- 格式转换要求
- Workspace 要求
- Epilogue 操作

### 实现
Agent: `ascendc-kernel-developer`
技能: `ascendc-direct-invoke-template`, `ascendc-blaze-best-practice`

### 性能
必须报告：
- TFLOPS（计算: MACs / primary_compute_kernel_us）
- MACs: 每次逻辑 MatMul 为 2 × M × N × K
- 格式转换开销
- Cube 利用率

**禁止：**
- Vector 逐元素乘冒充 MatMul
- ACLNN wrapper 冒充自定义 Cube
- 主机 MatMul
- 无 TFLOPS 的内核延迟
- 无形状边界文档的硬编码 tile

### Epilogue 处理
对于带 epilogue 的算子（如 MatMul 后的偏置加法、激活函数）：
- 在 Cube 流水线中记录 epilogue
- 使用 Fixpipe 进行 L0C→GM 转换
- 在性能测量中计入 epilogue 开销
- 不要将 epilogue 与主内核分开测量
