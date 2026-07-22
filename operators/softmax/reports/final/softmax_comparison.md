# Softmax 算子三路对比报告

## 算子信息
- 公式: `softmax(x, axis=-1) = exp(x - max) / sum(exp(x - max))`
- Shape: `[B, 256, 32]`, 沿最后一维 softmax
- FP16 输入, FP32 内部计算

## 正确性 (atol=0.01)

| 路由 | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|:---:|:---:|:---:|:---:|:---:|:----:|:----:|:----:|
| Torch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ascend C | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PyPTO | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## 性能 (event host_sync_us)

| Batch | Torch | Ascend C | PyPTO |
|:----:|:----:|:--------:|:-----:|
| 1 | 17.5 us | **6.8 us** | TBD |
| 2 | 17.5 us | 11.6 us | |
| 4 | 18.6 us | 21.4 us | |
| 8 | 18.4 us | 40.6 us | |
| 16 | 18.7 us | 78.8 us | |
| 32 | 18.0 us | 96.1 us | |
| 64 | 17.6 us | 193.3 us | |

## msprof (B=1)

| 路由 | primary_kernel_us | Kernel 类型 | 次数 |
|:----:|:----------------:|:----------:|:----:|
| Ascend C | **13.98 us** | KERNEL_AIVEC | 7 |
