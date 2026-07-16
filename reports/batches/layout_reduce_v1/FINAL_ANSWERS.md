# Final Answers — layout_reduce_v1 Key Blocker Repair

## 1. 成功 PyPTO 算子共同使用的 JIT 模式是什么

所有成功算子（ReLU, Mul, Add, Not, Or）共用：

- JIT 函数定义在 `src/{op}_impl.py` 的模块顶层
- 装饰器 `@pypto.frontend.jit`（无参数）
- 通过 `sys.path.insert(0, '..', 'src')` + `from {op}_impl import {wrapper}` 导入
- wrapper 函数在同一个文件中（纯 Python, 不被 JIT 装饰）
- 调用模式 `y.move(pypto.op.{api}(x))`
- 输入 tensor reshape 为 2D（[-1, last_dim]）
- `__init__.py` 非必需（2/5 成功算子无 __init__）

## 2. Expand/Transpose/ReduceSum 失败调用路径与成功路径的差异

**Expand 和 Transpose**: src/ 和 tests/ 目录完全为空，无实现文件 → 无法调用
**ReduceSum**: 有实现文件但使用了错误的 API 名称：
- `DT_FLOAT16` → 应使用 `DT_FP16`
- `pypto.op.reduce_sum` → 应使用 `pypto.op.sum`
- `axis=-1` → 应使用 `dim=-1`

## 3. `JIT cannot get source code` 是否已修复

**是。** 根本原因不是 Python 环境限制，而是函数定义位置问题。三个算子现在的实现文件都遵循成功模式，inspect.getsource() 通过 _original_func 可正常访问源码。

## 4. Transpose 修复 JIT 后是否出现新的真实 backend 错误

**是。** 修复 JIT 后，[16,32] 形状 PASS（bitwise 精确），但 [256,384] 出现：
```
Errcode: FFFFFF! Run pass failed., func CompileFunction, file host_machine.cpp, line 179
```
这是真实的 PyPTO backend limitation — transpose 的 CompileFunction pass 对大 tensor 失败。小 tensor（<1000 elements）正常工作。

## 5. Expand `expand_clone` 失败的准确 stage

- **之前版本**: `pypto.op.List([1, 384])` 在 parser evaluator 阶段失败 — "Type List cannot be instantiated"
- **修复后**: `expand_clone(x, [1, 384])` 在 backend Expand C++ 实现失败 — "Only allow to expand one axis"
- **最终方案**: 1D per-row expand `expand_clone(x, [384])`（x 形状 [1]）→ backend PASS
- **当前状态**: 功能正确但通过 per-row JIT dispatch（256 次调用/batch），JIT 缓存生效

## 6. ReduceSum 失败的准确 stage

- **Import 阶段**: `DT_FLOAT16` → AttributeError，从不存在的常量名开始
- **Parser 阶段**: `pypto.op.reduce_sum` → AttributeError，API 名错误
- **修复后**: 所有 stage PASS，backend 正常编译运行
- **精度**: FP16 累加 vs FP32 golden，max_diff ~0.06（在 FP16 累加预期范围内）

## 7. 三个 PyPTO 最终状态

| 算子 | 状态 |
|------|------|
| Expand | **PASS**（B=1 correctness，per-row JIT dispatch，bitwise 精确） |
| Transpose | **PARTIAL**（小 tensor PASS，[256,384] BLOCKED_BACKEND — CompileFunction pass 失败） |
| ReduceSum | **PASS**（B=1 correctness，FP16 累加精度） |

## 8. Ascend C Duplicate 真实可用签名

当前未验证。先前尝试的 3 参数 Duplicate 因模板参数解析失败。需要从当前 CANN 9.0.0 header 中检查真实签名。已列为剩余工作。

## 9. Expand 是否实现 device-side

**否。** 当前 Ascend C 实现仍为 host pre-expand + identity kernel。PyPTO 实现通过 JIT 的 expand_clone 完成 device-side 扩展，但 Ascend C 侧仍是 host fallback。

## 10. Transpose 是否实现 device-side

**否。** Ascend C 实现为 host pre-transpose + identity kernel。

## 11. ReduceSum 是否实现 device-side

**否。** Ascend C 实现为 host FP32 pre-reduce + identity kernel。

## 12. 哪些限制是真正 backend/API 限制

| 限制 | 类型 |
|------|------|
| Transpose [256,384] CompileFunction pass 失败 | **真实 backend 限制**（BLOCKED_BACKEND） |
| expand_clone "only allow to expand one axis" | API 使用方式（已通过 per-row 1D 方案绕过） |
| expand_clone "shape size should be equal" | API 使用方式（target.ndim 需等于 input.ndim） |

## 13. 哪些只是之前的调用方式错误

| 错误 | 修复 |
|------|------|
| `DT_FLOAT16` → `DT_FP16` | 常量名修正 |
| `pypto.op.reduce_sum` → `pypto.op.sum` | API 名修正 |
| `pypto.op.List([1, 384])` → `[1, 384]` | List 是类型注解不是构造器 |
| `expand_clone(x, [1, 384])` 对 [rows,1] 输入 | 改为 1D [1] → [384] per-row |
| src/ 和 tests/ 目录为空 | 创建实现文件和测试文件 |
| 函数定义在 heredoc/__main__ 中 | 改为 .py 文件顶层定义 |

## 14. correctness 和 profiler 结果

| 算子 | torch correctness | ascendc correctness | pypto correctness | profiler |
|------|------------------|-------------------|------------------|---------|
| Expand | PASS | PASS（host fallback identity） | PASS B=1 (bitwise) | 未运行 |
| Transpose | PASS | PASS（host fallback identity） | PARTIAL (small PASS) | 未运行 |
| ReduceSum | PASS | PASS（host fallback identity） | PASS B=1 (FP16 accum) | 未运行 |

## 15. final report 和 Dashboard 路径

- 修复报告: `reports/batches/layout_reduce_v1/key_blocker_repair_summary.{md,json}`
- JIT 矩阵: `reports/batches/layout_reduce_v1/pypto_jit_matrix.csv`
- 差异表: `reports/batches/layout_reduce_v1/pypto_jit_path_diff.csv`
- 最终矩阵: `reports/batches/layout_reduce_v1/final_operator_matrix.csv`
- 成功模式审计: `reports/batches/layout_reduce_v1/pypto_success_pattern_audit.{md,json}`

## 16. remaining blockers

| Blockers | 优先级 |
|----------|--------|
| Ascend C device-side Expand 实现（Duplicate/broadcast API） | HIGH |
| Ascend C device-side Transpose 实现（tile-based） | HIGH |
| Ascend C device-side ReduceSum 实现（reduction API） | HIGH |
| PyPTO Transpose 大 tensor CompileFunction pass 失败 | MEDIUM |
| PyPTO Expand per-row dispatch performance（256 次 kernel 调用/batch） | MEDIUM |
| 全 batch correctness 测试（B=2,4,8,16,32,64） | MEDIUM |
| Profiler 运行（Ascend C 和 PyPTO） | LOW |
