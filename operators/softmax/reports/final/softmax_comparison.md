# Softmax 算子三路对比报告

## 算子信息
- 公式: `softmax(x, axis=-1) = exp(x - max) / sum(exp(x - max))`
- Shape: `[B, 256, 32]`, 沿最后一维 softmax
- FP16 输入, FP32 内部计算

## 正确性 (atol=0.01, rtol=0.001)

| 路由 | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|:---:|:---:|:---:|:---:|:---:|:----:|:----:|:----:|
| Torch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ascend C | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PyPTO | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## 性能

### Event host_sync_us (中位数, warmup=200, loops=100, repeat=5)

| Batch | Torch | Ascend C | 加速比 |
|:----:|:----:|:--------:|:-----:|
| 1 | 16.0 us | **6.8 us** | 2.4x |
| 2 | 16.7 us | 11.6 us | 1.4x |
| 4 | 16.8 us | 21.3 us | 0.8x |
| 8 | 16.6 us | 40.6 us | 0.4x |
| 16 | 16.8 us | 78.8 us | 0.2x |
| 32 | 16.4 us | 96.0 us | 0.2x |
| 64 | 16.0 us | 193.5 us | 0.08x |

### msprof primary_compute_kernel_us

| Batch | Ascend C (KERNEL_AIVEC) |
|:----:|:----------------------:|
| 1 | 13.98 us |
| 32 | 108.9 us |
| 64 | 209.5 us |

**分析**: Torch 的 softmax 高度优化（B=64 仅 16 us），Ascend C 在 B=1 时 2.4x 领先，但 B≥4 后被反超。原因：逐行串行 pipeline 固定开销大，Torch 的 batch 并行度高。
