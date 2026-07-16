# MatMul Operator SPEC

## 算子名称
matmul

## 数学公式
Y[b,h,:,:] = A[b,h,:,:] @ B[b,h,:,:]

## 输入输出规格
- A: [B,12,256,256], FP16, ND contiguous
- B: [B,12,256,32], FP16, ND contiguous
- Y: [B,12,256,32], FP16, ND contiguous

## 语义
对于每个 batch b 和 head h，计算矩阵乘法：
Y[b,h,:,:] = A[b,h,:,:] @ B[b,h,:,:]

其中单矩阵规格:
- M = 256, K = 256, N = 32

## 精度要求
- atol: 0.01
- rtol: 0.01
- require_bitwise: false
- 注意: FP16 MatMul 使用 FP16 累加，与 FP32 累加结果有差异

## 动态轴
- B: batch size, [1, 2, 4, 8, 16, 32]
