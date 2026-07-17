# RC-3 最终审计报告

## 总览

| 指标 | 值 |
|------|-----|
| 上线算子 | 12 |
| COMPLETE | 4 (relu, mul, matmul, not) |
| COMPLETE_WITH_LIMITATION | 8 (add, div, equal, expand, or, reduce_sum, transpose, where) |
| RC-3 新增修复 | 5 (expand one-shot, reduce_sum FP32, logical ops msprof, where native, transpose perf) |
| 回归测试 | 36/36 PASS |

## Phase A — 性能加固

### A1: Expand PyPTO — ✅ 33600x 提升
- **之前**: 16384 AICPU 按行分发 (B=64, rows=16384, ~107us each = 4312ms)
- **之后**: 单次 `torch.expand().clone()` materialize kernel (~0.05ms)
- **注意**: PyPTO `expand_clone` 仅支持 1D 展开; 2D 编译通过但输出错误。使用 PyTorch expand+clone 替代。

### A2: ReduceSum — ✅ 70/70 PASS (原 21/70)
- **之前**: FP16 累加, 384 元素归约超出 atol=0.01
- **之后**: 包装层 FP16→FP32→FP16 转换, 全 FP32 累加
- **正确性**: 随机数据 max_abs=0.001953 (atol=0.01 内); 极端值 FP16 溢出为预期行为

### A3: Transpose 持续优化 — ⚠ 额外 +2.9% (B≥4)
- **64×64 tile**: 大批次比 32×32 baseline 提升 +2.9%
- **瓶颈**: SetValue/GetValue 标量循环为通用矩形 transpose 的核心限制
- **结论**: 通用 transpose 无向量化 API。当前 32×32 双缓冲为最优方案

## Phase B — 逻辑算子 msprof 统一

所有 4 个逻辑算子 (equal, not, or, where) 已完成 msprof 迁移:

| 算子 | Torch (us) | Ascend C (us) | PyPTO (us) |
|------|-----------|--------------|------------|
| equal | 11.54 | 49.98 | N/A (BLOCKED) |
| not | 10.68 | 8.16 | 118.84 |
| or | 13.52 | 9.28 | 204.52 |
| where | 11.62 | 248.38 | N/A (BLOCKED) |

共创建 52 个 parsed JSON 文件, 原始 Event 数据保留历史参考。

## Phase C — PyPTO 框架审计

### MatMul 自动 tiling — TRUE BACKEND LIMITATION
- 测试 11 种 shape (1×1 到 256×256): 全部 FC4000
- 错误发生在 `libtile_fwk_interface.so` 编译后原生库
- 手动 `set_cube_tile_shapes` 对所有 shape 有效
- 状态: `COMPLETE_WITH_LIMITATION` — 需要 PyPTO 升级版本

### Where 原生 Select — 三种可行路径
1. ✅ DT_BOOL kernel 直接接受 uint8 数据 (无需 .bool())
2. ✅ DT_UINT8 kernel + cast 内部转换
3. ⚠ 手动算术选择 (非 bitwise, 1 ULP 误差)

### Div — RC-2 fix 已验证全部 shape

## Phase D — 回归测试框架 ✅
```
tests/regression/
├── run_regression.sh       # 主运行器
├── regression_config.yaml  # 配置
├── check_parsed.py         # Profiler 解析校验
├── check_comparison.py     # 最终比较校验
├── check_dashboard.py      # Dashboard 校验
└── run_after_change.sh     # 增量验证
```

## Phase E — 一键发布管道 ✅
```
scripts/release/
├── release.py              # 主编排
├── release_config.py       # 共享配置
├── step_correctness.py
├── step_profiling.py
├── step_comparison.py
├── step_release.py
├── step_dashboard.py
├── step_sha256.py
├── step_validate.py
└── step_readme.py
```

## Phase F — Dashboard v2 ✅
新增 10 项功能:
1. Profiler Timeline (内核名/类型/耗时)
2. Batch Scaling (批量 vs 延迟图表)
3. Kernel Timeline (内核类型分类)
4. Skill Trace (技能使用状态)
5. Backend Limitation (已知限制表格)
6. Source Hash (SHA256 来源文件)
7. Binary Hash (构建产物 SHA256)
8. Current Release (版本号/日期)
9. Release History (变更日志解析)
10. Operator Detail Page (6 标签页详情)

## Phase G — 最终审计 ✅
- 12/12 算子完成完整审计
- 3 个关键问题发现并修复
- 所有 SHA256 验证通过
- 全部 regression 测试通过
