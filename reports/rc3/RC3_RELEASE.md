# RC-3 Release Notes — v1.3-rc3

## 新功能

### 一键发布 Pipeline
```bash
python3 scripts/release/release.py
```
自动按序执行: correctness → profiling → parser → comparison → release → dashboard → README → SHA256 → validation

### 回归测试框架
```bash
bash tests/regression/run_regression.sh
```
包含 5 项检查: SHA256, build, correctness, profiler, parsed validation。支持 --skip-* 和 --json 输出。

### Dashboard v2
10 项新功能: profiler timeline, batch scaling, skill trace, backend limitation, source hash, release history, operator detail page

## 修复

| # | 算子 | 路线 | 问题 | 修复 |
|---|------|------|------|------|
| 1 | expand | PyPTO | 16384 AICPU 按行分发 → 4312ms | 单次 torch.expand().clone() → ~0.05ms (33600x) |
| 2 | reduce_sum | PyPTO | FP16 累加 → 21/70 PASS | FP32 包装累加 → 70/70 PASS |
| 3 | equal | msprof | Event 仅历史 | msprof 统一, 52 个 parsed 文件 |
| 4 | not | msprof | Event 仅历史 | msprof 统一, 21 个 parsed 文件 |
| 5 | or | msprof | Event 仅历史 | msprof 统一, 21 个 parsed 文件 |
| 6 | where | msprof | Event 仅历史 | msprof 统一, 14 个 parsed 文件 |
| 7 | transpose | Ascend C | 性能提升 | 64×64 tile +2.9% (B≥4) |
| 8 | where | PyPTO | .bool() 包装层冗余 | DT_BOOL kernel 直接接受 uint8 |
| 9 | 全部 | SHA256 | 11/12 算子校验失败 | 全部重建验证通过 |
| 10 | 全部 | 审计 | 3 个不一致 | profiler 分类/README/changelog 修复 |

## 已知限制 (Remaining)

| 优先级 | 算子 | 路线 | 描述 |
|--------|------|------|------|
| P1 | or | PyPTO | 使用 bitwise_or (无 logical_or API) |
| P1 | reduce_sum | 全部 | FP16 输出溢出 >65504 |
| P2 | matmul | PyPTO | 自动 tiling FC4000, 需手动 tile |
| P2 | equal | PyPTO | BOOL 输出需 ta≤64 tile 约束 |
| P2 | where | PyPTO | uint8 condition → DT_BOOL 转换 |
| P2 | expand | PyPTO | 使用 PyTorch 而非原生 expand_clone |
| P2 | add | PyPTO | B=2..64 正确性未持久化 |
