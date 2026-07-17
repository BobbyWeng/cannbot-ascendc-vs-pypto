# Vector 算子工作流（中文版）

适用于 `hardware_path: VECTOR` 的算子。

示例算子：relu, add, mul, div, not, equal, where, reduce_sum。

## 工作流

与父工作流（见 `cn_OPERATOR_DEVELOPMENT_WORKFLOW.md`）相同，增加 vector 特定内容：

### 前置检查
- 加载 `ascendc-env-check` 技能
- 验证 NPU 设备可用
- 验证 CANN 编译器可用

### 设计（Ascend C）
Agent: `ascendc-kernel-architect`
技能: `ascendc-kernel-develop-workflow`, `ascendc-tiling-design`, `ascendc-api-best-practices`, `ascendc-docs-search`

必须记录：
- GM↔UB 流水线设计
- 队列/缓冲区配置
- Vector API 选择（DataCopy, Add, Sub, Mul 等）
- Mask/repeat 策略
- 对齐要求
- 尾部处理（剩余元素）
- blockDim 计算
- TotalElements → tile 计数
- Batch 跨度
- 双缓冲方案
- 标量回退（若有）
- 预期有效带宽

### 实现
Agent: `ascendc-kernel-developer`
技能: `ascendc-direct-invoke-template`, `ascendc-api-best-practices`

必须使用：
- 原生 Ascend C Vector API（非 PyPTO）
- `<<<>>>` 直接调用模式
- 模板中的 CMakeLists.txt
- 设计中的 Tiling 头文件

### 正确性
- 所有正式批次必须通过
- FP16: 对比 FP32 torch 参考
- FP32: 位精确或规格定义容差
- 归约（如 reduce_sum）: 记录累积 dtype

### 性能
- 主要指标: `primary_compute_kernel_us`
- 带宽: 字节数 / primary_compute_kernel_us
- 必须报告有效带宽

## 模板结构

```
operators/{op}/
├── ascendc/        # Ascend C 内核代码
├── torch/          # Torch 正确性和 benchmark
├── pypto/          # PyPTO（若适用）
├── data/           # 测试数据
├── benchmark/      # 基准测试
├── reports/        # 报告
├── SPEC.yaml       # 规格
├── TASK_STATE.json # 状态
├── SKILL_TRACE.md  # 技能跟踪
├── README.md       # 实现说明
├── REPRODUCE.md    # 复现步骤
└── SHA256SUMS      # 哈希
```
