# MatMul Cube 算子设计文档

## 0. 概述

### 0.0 需求类型
- **特定用例**：A [B,12,256,256] × B [B,12,256,32] → Y [B,12,256,32], FP16

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | matmul |
| 算子类别 | MatMul (Cube) |
| 需求类型 | 特定用例 |
| 支持数据类型 | FP16 |
| 支持服务器 | A3 (Ascend 910B) |
| 特殊约束 | N=32 较小，Cube 利用率需关注 |

### 0.2 算子类别识别
- **类别**：MatMul 矩阵乘类
- **判断依据**：矩阵乘法，高计算密度，使用 Cube 单元

### 0.3 成熟方案查阅
- **是否查阅成熟方案**：是
- **参考文档**：
  - `/mnt/workspace/asc-devkit/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul/` — 基础 MatMul 示例
  - `/mnt/workspace/asc-devkit/examples/01_simd_cpp_api/04_advanced_api/00_matmul/batch_matmul/` — Batch MatMul 示例
  - `/mnt/workspace/asc-devkit/examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_high_performance/` — 高性能 MatMul
  - `/mnt/workspace/asc-devkit/docs/zh/api/SIMD-API/高阶API/矩阵计算/Matmul-Kernel侧接口/` — Matmul API 文档

### 0.4 应用关键设计

| 设计项 | 成熟方案 | 应用到当前算子 |
|--------|---------|----------------|
| Matmul 高阶 API | `AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM>` | 使用高阶 API，不需要手写 MMAD |
| REGIST_MATMUL_OBJ | 注册宏 + tiling | 使用 `REGIST_MATMUL_OBJ(&pipe, workspace, matmulObj, &tiling)` |
| SetTensorA/SetTensorB | 设置 GM tensor | 每个矩阵独立设置 |
| IterateAll | 一次计算完成 | `matmulObj.IterateAll(cGlobal)` |
| 多核切分 | 按 batch/head 切分 | 每个 core 处理多个独立矩阵 |

## 1. 算子设计

### 1.1 数学公式

```
输入: A [B,12,256,256], FP16
      B [B,12,256,32], FP16
输出: Y [B,12,256,32], FP16

公式: Y[b,h,:,:] = A[b,h,:,:] @ B[b,h,:,:]
单矩阵: M=256, K=256, N=32
总矩阵数: B × 12
FLOPs/矩阵: 2 × 256 × 32 × 256 = 4,194,304
```

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 |
|---------|---------|---------|---------|
| 矩阵乘 | `AscendC::Matmul` | `aType, bType, cType, biasType, CFG_NORM` | ND |
| 设置 A | `SetTensorA(aGlobal, false)` | `aGlobal`, 不转置 | ND→Cube 内部格式 |
| 设置 B | `SetTensorB(bGlobal, false)` | `bGlobal`, 不转置 | ND→Cube 内部格式 |
| 计算 | `IterateAll(cGlobal, 0)` | `cGlobal`, 非原子 | Cube MMAD |
| Tiling 注册 | `REGIST_MATMUL_OBJ` | pipe, workspace, obj, tiling | - |

#### 1.2.1 API 语义验证

| API | 数据布局 | 功能需求 | 限制条件 | 匹配 |
|-----|---------|---------|---------|------|
| Matmul ND | A[M,K] ND, B[K,N] ND, C[M,N] ND | 矩阵乘 FP16→FP16 | M/K/N ≥ 16 对齐 | ✅ |
| SetTensorA | GlobalTensor<half> ND | A 矩阵输入 | 连续 GM 地址 | ✅ |
| IterateAll | GlobalTensor<half> ND 输出 | 完整矩阵计算 | 单核单矩阵 | ✅ |

### 1.3 数据流

```
输入 A (GM, ND) ──→ SetTensorA(aGmTensor, false)
                       ↓ Cube MMAD (内部 GM→L1→L0A→MMAD→L0C→Fixpipe→GM)
输入 B (GM, ND) ──→ SetTensorB(bGmTensor, false)
                       ↓ IterateAll(cGmTensor, 0)
输出 Y (GM, ND) ←────┘
```

### 1.4 核心计算步骤

```
循环: 对每个 core 分配的矩阵列表
  1. 计算 GM 地址偏移: aOff = matIdx × M × K
  2. 构造 GlobalTensor<half> aGm(aBase + aOff, M × K)
  3. SetTensorA(aGm, false)
  4. 同理 SetTensorB(bGm, false)
  5. REGIST_MATMUL_OBJ + Init
  6. IterateAll(cGm, 0)
```

### 1.5 Tiling 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| M | 256 | A 行数 |
| N | 32 | B 列数 |
| K | 256 | 内维度 |
| baseM | 256 | 单核 M |
| baseN | 32 | 单核 N |
| baseK | 256 | 单核 K |
| depthA1 | 2 | Double buffer A |
| depthB1 | 2 | Double buffer B |
| iterateOrder | 0 | 默认迭代顺序 |

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | 按独立矩阵 (batch × head) |
| 单核任务量 | ceil(totalMatrices / blockDim) |
| 使用的核数 | `ACL_DEV_ATTR_CUBE_CORE_NUM` 动态获取 |
| 负载均衡 | 均匀分配，尾核处理剩余矩阵 |

### 2.2 Tiling 设计

使用 Matmul 高阶 API 自动 tiling：通过 `REGIST_MATMUL_OBJ` 注册 tiling 参数，API 内部自动计算 L1/L0A/L0B/L0C 切分。

### 2.3 Pipeline

```
REGIST_MATMUL_OBJ → Init → SetTensorA → SetTensorB → IterateAll → End

Cube 内部流水线 (自动管理):
  GM─→L1 (DataCopy GM→L1)
  L1─→L0A (LoadData)
  L1─→L0B (LoadData)
  L0A × L0B ─→ MMAD ─→ L0C
  L0C ─→ Fixpipe ─→ GM (写入 C)
```

## 3. 实施计划

### 3.1 文件清单

| 文件 | 路径 |
|------|------|
| Kernel | `operators/matmul/ascendc/src/matmul_kernel.asc` |
| Host | `operators/matmul/ascendc/src/matmul_host.asc` |
| Tiling | `operators/matmul/ascendc/src/matmul_tiling.h` |
| CMake | `operators/matmul/ascendc/CMakeLists.txt` |

### 3.2 测试计划

- B=1,2,4,8,16,32 全 batch 正确性
- 与 Torch FP16 和 FP32 reference 对比
- Tolerance: atol=0.03125, rtol=1.5

### 3.3 风险评估

| 风险 | 可能性 | 缓解措施 |
|------|--------|---------|
| Matmul API 需要 tiling 库 | 中 | 使用 `REGIST_MATMUL_OBJ` 自动处理 |
| N=32 利用率低 | 高 | 按 head 融合 — 后续优化 |
