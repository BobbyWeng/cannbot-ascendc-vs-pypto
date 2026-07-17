# Cannbot: Ascend C vs PyPTO 算子对比项目 — v1.3-rc3

在昇腾 NPU 硬件上对 Ascend C 和 PyPTO 算子实现与 torch 基线进行结构化、可复现的对比评测框架。

## 项目结构

```
cannbot_ascendc_vs_pypto/
├── cn_README.md                    # 本文件（中文版）
├── README.md                       # 英文版
├── AGENTS.md                       # Agent 配置（英文）
├── cn_AGENTS.md                    # Agent 配置（中文）
├── environment/                    # 环境和版本清单
├── common/                         # 共享库（profiler、correctness、benchmark）
├── operators/                      # 12 个算子对比目录
│   ├── relu/                       # ReLU（COMPLETE）
│   ├── mul/                        # 乘法（COMPLETE）
│   ├── add/                        # 加法（COMPLETE_WITH_LIMITATION）
│   ├── div/                        # 除法（COMPLETE_WITH_LIMITATION）
│   ├── equal/                      # 等于（COMPLETE_WITH_LIMITATION）
│   ├── not/                        # 逻辑非（COMPLETE）
│   ├── or/                         # 逻辑或（COMPLETE_WITH_LIMITATION）
│   ├── where/                      # 条件选择（COMPLETE_WITH_LIMITATION）
│   ├── expand/                     # 扩展（COMPLETE_WITH_LIMITATION）
│   ├── transpose/                  # 转置（COMPLETE_WITH_LIMITATION）
│   ├── reduce_sum/                 # 归约求和（COMPLETE_WITH_LIMITATION）
│   └── matmul/                     # 矩阵乘（COMPLETE）
├── templates/                      # 新算子模板
├── reports/                        # 项目级报告
│   ├── release/                    # 当前发布（单一事实源）
│   │   ├── current_release.json    # 机器可读发布状态
│   │   ├── current_release.md      # 人类可读发布摘要
│   │   ├── operator_matrix.csv     # 算子状态矩阵
│   │   ├── performance_matrix.csv  # 性能对比
│   │   ├── correctness_matrix.csv  # 正确性覆盖
│   │   ├── limitation_matrix.md    # 已知限制
│   │   └── limitation_matrix.json  # 机器可读限制
│   ├── rc3/                        # RC-3 最终报告
│   └── operator_summary.md/json    # 快速参考摘要
├── dashboard/                      # 独立仪表盘
├── scripts/                        # NPU 锁、profiler 队列脚本
└── archives/                       # 算子归档包
```

## 算子状态 (v1.3-rc3)

| 算子 | 最终状态 | Torch | Ascend C | PyPTO | 正确性 | Profiler |
|------|---------|-------|----------|-------|--------|----------|
| relu | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | 全部 (7/7) | msprof |
| mul | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | 全部 (7/7) | msprof |
| not | **COMPLETE** | PASS | TRUE_DEVICE | SUCCESS | 全部 (42/42) | msprof |
| matmul | **COMPLETE** | PASS | TRUE_CUBE | SUCCESS(受限) | 全部3路线 (6/6) | msprof |
| add | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | PyPTO B=1 持久化 | msprof |
| div | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | 全部3路线 (6/6 位精确) | msprof |
| equal | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | 全部3路线 (7/7 位精确) | msprof |
| or | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | BITWISE_OR | AscendC 修正 (49/49) | msprof |
| where | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | 全部3路线 (7/7 位精确) | msprof |
| expand | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | 全部 (7/7) | msprof |
| transpose | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | 全部3路线 (7/7 位精确) | msprof |
| reduce_sum | **COMPLETE_WITH_LIMITATION** | PASS | TRUE_DEVICE | SUCCESS | 70/70 (FP32累积) | msprof |

**状态**: 4 个 COMPLETE, 8 个 COMPLETE_WITH_LIMITATION, 0 个 PARTIAL, 0 个 INCOMPLETE

## 测量方法

所有 msprof 算子使用 **统一 profiler 测量**：
1. **Profiler**: msprof 参数 `--ascendcl=on --ai-core=on --task-time=l0`
2. **预热**: 200 次迭代（不计入测量）
3. **测量迭代**: 100 次迭代
4. **关键指标**: `primary_compute_kernel_us` — 每次调用中最长的单个设备内核事件
5. **PyPTO**: 双进程方法（预热无 profiler，然后 msprof）以排除 JIT 编译

## 已知限制

| 优先级 | 算子 | 路线 | 描述 |
|-------|------|------|------|
| P1 | or | PyPTO | 使用 bitwise_or（无 logical_or API）。对 0/1 bool 正确。 |
| P1 | reduce_sum | 全部 | FP16 输出溢出 >65504（预期行为） |
| P2 | matmul | PyPTO | 自动 tiling FC4000；需要手动 set_cube_tile_shapes |
| P2 | equal | PyPTO | BOOL 输出需要 ta≤64 tile 约束 |
| P2 | where | PyPTO | uint8 条件需要 DT_BOOL kernel |
| P2 | expand | PyPTO | 使用 PyTorch expand+clone（非 PyPTO 原生） |
| P2 | add | PyPTO | 正确性 B=2..64 未持久化到 JSON |

## 仪表盘

```bash
python dashboard/dashboard.py --release reports/release/current_release.json
```

用浏览器打开 `dashboard/index.html`。

## 环境要求

- Ascend 910B 或兼容 NPU
- CANN Toolkit（已验证 9.0.0）
- Python 3.8+
- PyTorch + torch_npu
- PyPTO 框架
- Cannbot Skills（用于 Ascend C 开发）

## 复现结果

参见 `reports/rc3/` 查看 RC-3 最终审计报告。历史报告在 `archives/rc_history/`。

参见 `operators/{op}/REPRODUCE.md` 查看每个算子的逐步指南。
