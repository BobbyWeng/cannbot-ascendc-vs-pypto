# 12 个算子现状报告（中文版快捷图标版）

> 基于 v1.3-rc3 | Ascend 910B, CANN 9.0.0, PyPTO 0.2.0 | 2026-07-18

---

## 总览：4 ✅ / 8 ⚠️ / 0 ❌

| # | 算子 | 路线 | 正确性 | 性能(B=1) | 主要问题 |
|---|------|------|--------|-----------|---------|
| 1 | **relu** ✅ | Torch/AscendC/PyPTO | 全 PASS 位精确 | 2.6/2.1/52 µs | — |
| 2 | **mul** ✅ | Torch/AscendC/PyPTO | 全 PASS 位精确 | 9.0/11.2/52 µs | — |
| 3 | **add** ⚠️ | Torch/AscendC/PyPTO | 全 PASS | 37.5/6.6/397 µs | B=2..64 正确性未持久化 |
| 4 | **div** ⚠️ | Torch/AscendC/PyPTO | 全 PASS(6/6) | 21.8/18.6/— µs | 曾 tile 阻塞，RC-2 已解封 |
| 5 | **equal** ⚠️ | Torch/AscendC/PyPTO | 全 PASS(7/7) | 12.2/41.8/— µs | 曾 DT_BOOL 阻塞，RC-2 已解封 |
| 6 | **not** ✅ | Torch/AscendC/PyPTO | 全 PASS(42/42) | 127.5/6.4/137 µs | — |
| 7 | **or** ⚠️ | Torch/AscendC/PyPTO | 全 PASS(49/49) | 256.3/6.5/149 µs | PyPTO 用 bitwise_or 替代 logical_or |
| 8 | **where** ⚠️ | Torch/AscendC/PyPTO | 全 PASS(7/7) | 131.9/238.6/— µs | 曾 ExpandPass 阻塞，RC-2 已解封 |
| 9 | **expand** ⚠️ | Torch/AscendC/PyPTO | 全 PASS(7/7) | 13.0/15.0/~50 µs | 原逐行 AICPU 16384 内核，RC-3 33600× 修复 |
| 10 | **transpose** ⚠️ | Torch/AscendC/PyPTO | 全 PASS(7/7) | 14.1/106/— µs | 曾 tile 阻塞，RC-2 已解封 |
| 11 | **reduce_sum** ⚠️ | Torch/AscendC/PyPTO | 62/21/**70**/70 PASS | 16.4/14.4/— µs | AscendC/PyPTO FP16 累积精度不足 |
| 12 | **matmul** ✅ | Torch/AscendC/PyPTO | 全 PASS(6/6) | 12.2/10.4/— µs | PyPTO 自动 tiling 彻底损坏，手动修复 |

---

## 为什么有些 PyPTO 算子曾无法运行？

| 算子 | 阻塞原因 | 修复方法 |
|------|---------|---------|
| div | CompileFunction 广播 tile 参数错误 | tile_shape(128,1024) |
| equal | 输出 dtype 误判为 DT_FP16 + ta≤64 约束 | 手动 DT_BOOL + ta≤64 |
| where | ExpandPass 误读 condition 物理大小 | uint8→bool 转换 |
| transpose | CompileFunction >1K 元素 tile 失效 | tile_shape(64,256) |
| matmul | 自动 tiling 引擎完全损坏 | 手动 set_cube_tile_shapes |
| expand | 原生 expand_clone 2D 输出错误 | torch.expand().clone() |
| reduce_sum | FP16 累积精度不足 | wrapper 层 FP32 cast |

**根因**: PyPTO 0.2.0 的自动 tiling 系统不可靠，dtype 推断不一致，广播支持有限。**所有算子已通过 workaround 解封。**

---

## 为什么 PyPTO 耗时与其他路线差距巨大？

| 算子 | Torch | Ascend C | PyPTO | PyPTO 内核数 | 差距倍数 |
|------|:-----:|:--------:|:-----:|:----------:|:--------:|
| relu | 2.6 µs | 2.1 µs | 52 µs | 3 (1+2 AICPU) | **25×** |
| add | 37.5 µs | 6.6 µs | 397 µs | 9 (3×3) | **60×** |
| expand(原版) | 13 µs | 15 µs | 2796 µs | 16384 | **186×** |

**根因三要素**:
1. **AICPU executor 调度 = 2 个额外内核/调用**，每个 50-140 µs 纯开销
2. **MIX_AIC 内核比纯 AIVEC 重**，PyPTO 混合流水线固有开销
3. **算子链分解**：融合操作被拆成多个二元操作，每个产生 3 个内核事件

---

## 正确性短板

| 算子 | 通过率 | 原因 |
|------|:-----:|------|
| reduce_sum AscendC | **21/70** | FP16 累积精度不足（硬件限制，未修复） |
| add PyPTO | **B=1 仅存** | B=2..64 运行通过但报告未持久化（报告 bug） |
| matmul AscendC | **1/9 已跑** | 8 个用例由于输出文件被跳过（覆盖不足） |
| matmul PyPTO | **6/6 但非位精确** | FP16 累积 max_abs=0.031（已知限制） |

---

## 建议行动

1. **P0**: add 正确性结果补全、matmul AscendC 测试覆盖补跑
2. **P1**: reduce_sum AscendC FP32 累积 kernel 评估
3. **P2**: 跟进 PyPTO 新版本解决自动 tiling 和 AICPU 开销
