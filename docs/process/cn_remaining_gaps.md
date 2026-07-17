# 剩余流程缺口（中文版）

本文档记录流程重构中识别出的需要后续跟进的缺口。

## 缺口 1：现有算子缺少 TASK_STATE.json

**严重性**: 高
**描述**: 所有 12 个现有算子缺少 `TASK_STATE.json`。新工作流需要此文件用于恢复和门禁跟踪。
**行动**: 创建迁移脚本，通过检查现有工件为每个算子生成 TASK_STATE.json。

## 缺口 2：NPU 运行队列未填充

**严重性**: 中
**描述**: `reports/runtime/npu_run_queue.json` 不存在。NPU 调度是临时性的，没有序列化。
**行动**: 将 NPU 队列集成到 benchmark 脚本中。队列文件结构已定义但未强制执行。

## 缺口 3：现有算子未通过代码审查门禁 (G13)

**严重性**: 高
**描述**: 没有现有算子通过代码审查。任何算子都不存在 REVIEW.md 工件。
**行动**: 在下一次发布前对现有算子运行代码审查。这是发布前的阻塞项。

## 缺口 4：正确性报告目录缺失

**严重性**: 中
**描述**: 某些算子的 `operators/{op}/reports/correctness/` 目录存在但为空。正确性结果位于 `torch/` 或 `pypto/` 目录中。
**行动**: 将正确性报告位置标准化为 `operators/{op}/reports/correctness/`。

## 缺口 5：缺省权限延迟文件

**严重性**: 低
**描述**: `reports/runtime/permission_deferred.json` 不存在。当 external_directory 权限被阻塞时需要此文件。
**行动**: 创建默认空数组的文件。

## 缺口 6：Event 与 msprof 跨比较未解决

**严重性**: 高
**描述**: `not`、`equal`、`or`、`where` 等算子使用 `torch.npu.Event`（主机同步）而不是 msprof。新 AGENTS.md 明确禁止使用不同测量级别进行跨路线排名。
**行动**: 使用 msprof 重新 profile 这些算子，或在发布中永久标记为 NOT_COMPARABLE。

## 缺口 7：Cube 算子 TFLOPS 未计算

**严重性**: 中
**描述**: 现有 Cube 审计不计算 matmul 的 TFLOPS。新 AGENTS.md 要求 TFLOPS = MACs / kernel_us。
**行动**: 为 Cube 算子添加 TFLOPS 计算到解析流水线。

## 缺口 8：插件初始化未自动化

**严重性**: 低
**描述**: 新工作流需要加载正确的插件。虽然 `ops-direct-invoke` 和 `pypto-op-orchestrator` 已安装，但缺少自动步骤来验证为每个任务加载了正确的插件。
**行动**: 将插件验证添加到 G2 门禁检查。

## 缺口 9：G8 覆盖阻止未自动化

**严重性**: 高
**描述**: AGENTS.md 声明 G8 绝不可覆盖，但没有机械强制执行——只有文档中的文本。
**行动**: 在 `check_gate.py` 中添加明确的 G8 覆盖阻止，任何试图绕过 G8 的操作都会报错。

## 缺口 10：历史审计数据不能输入仪表盘

**严重性**: 中
**描述**: 新 AGENTS.md 声明历史审计不是仪表盘的输入。当前的 `reports/release/current_release.json` 可能包含过时数据。
**行动**: 验证 current_release.json 是否与实际工件状态一致。
