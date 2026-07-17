# 算子开发工作流（中文版）

本文档描述本项目标准算子开发工作流。父文档；路线特定详情在单独文件中。

## 通用流程

```
用户请求 → CLASSIFY(确定后端/语义/硬件/模式) → PLUGIN_SELECTION(选择插件)
→ PREFLIGHT(G0 环境检查) → SPEC(G3 算子规格) → API_ANALYSIS(G4 可行性)
→ GOLDEN/DATA(G5 参考实现+测试数据) → DESIGN(G6 架构/切分/流水线)
→ IMPLEMENTATION(G7 内核代码/构建) → CORRECTNESS(G8 硬门禁)
→ PASS: BASELINE_PROFILE(G10) → OPTIMIZATION(G11) → FINAL_PROFILE(G12)
→ FAIL: PRECISION_FIX(G9) → 重新验证
→ CODE_REVIEW(G13) → RELEASE(G14) → ARCHIVE(G15)
```

## 并发规则

- **串行（NPU）**: 正确性运行、PyPTO JIT、Ascend C 运行时、benchmark、msprof、候选项计时
- **并行（非NPU）**: 文档分析、源码研究、API 搜索、解析、报告生成
- **序列化工具**: `reports/runtime/npu_run_queue.json`

## 工件要求

参见 cn_AGENTS.md §17 获取完整工件列表。核心原则：每个阶段在下一阶段开始前必须产生可验证的工件。
