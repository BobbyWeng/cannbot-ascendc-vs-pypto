# Cube Operator Template v2

首次 Cube 类算子（Arithmetic / GEMM / Attention）统一模板。

## 与 Vector Template v1 的区别

| 维度 | Vector Template v1 | Cube Template v2 |
|------|-------------------|------------------|
| 计算单元 | AIV/Vector (SIMD) | AIC/Cube (MMAD) |
| 数据路径 | GM → UB → Vector → UB → GM | GM → L1 → L0A/L0B → MMAD → L0C → Fixpipe → GM |
| 主要 API | `AscendC::Add`, `DataCopyPad` | `Mmad`, `LoadData`, `Fixpipe`, `DataCopy` GM↔L1 |
| Tiling | 连续元素 | M/N/K 分块 + Block/Warp/Tile |
| 适用算子 | Add, Mul, Relu, Div | MatMul, BatchMatMul, Linear, QK, PV, FFN, Gemm |
| 流水线 | 单级 Vector 流水线 | 多级 GM→L1→L0A/B→MMAD→L0C→Fixpipe |
| 数据格式 | ND contiguous | ND/NZ, 支持 format conversion |

## GM/L1/L0/Cube 流水线

```
GM A (ND/NZ) ──→ L1 (通过 DataCopy GM→L1)
                      │
                      ├──→ L0A (通过 LoadData L1→L0A)
                      │
GM B (ND/NZ) ──→ L1 (通过 DataCopy GM→L1)
                      │
                      └──→ L0B (通过 LoadData L1→L0B)
                             │
                             ▼
                          Cube MMAD
                             │
                             ▼
                           L0C
                             │
                             ▼
                        Fixpipe / DataCopy L0C→GM
                             │
                             ▼
                          GM Y
```

## Tiling 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| M | uint32_t | A rows per matrix |
| N | uint32_t | B cols per matrix |
| K | uint32_t | inner dim per matrix |
| BASE_M | uint32_t | M 分块基数 |
| BASE_N | uint32_t | N 分块基数 |
| BASE_K | uint32_t | K 分块基数 |
| BLOCK_M | uint32_t | M tile size |
| BLOCK_N | uint32_t | N tile size |
| BLOCK_K | uint32_t | K tile size |
| batchCount | uint32_t | batch × head |
| blockDim | uint32_t | 总 AI Core 数 |
| matricesPerCore | uint32_t | 每个 core 分配的矩阵数 |
| transposeA | bool | A 是否转置 |
| transposeB | bool | B 是否转置 |
| inputDtype | uint32_t | 输入数据类型 |
| accumDtype | uint32_t | 累加数据类型 |
| outputDtype | uint32_t | 输出数据类型 |
| layoutA | uint32_t | A 数据格式 (ND=0, NZ=1) |
| layoutB | uint32_t | B 数据格式 |
| workspaceSize | uint32_t | workspace 字节数 |

## 数据格式

- 输入: FP16 ND 格式
- Cube 输入: 通常需要 NZ 格式或特定 Cube layout
- 输出: FP16 ND 格式

## 复用到其他算子

### BatchMatMul
- M, N, K 不变
- batchCount = B × heads
- 按 head 分 core

### Linear
- M = batch × seq_len
- N = hidden_out
- K = hidden_in

### QK
- M = seq_len_q
- N = seq_len_k
- K = head_dim
- 注意 softmax 参与

### PV
- M = seq_len_q
- N = head_dim_v
- K = seq_len_v

### FFN (up/gate/down)
- 三个独立 MatMul: up/gate/down
- FFN up: [B,S,4H] = [B,S,H] @ [H,4H]
- FFN gate: [B,S,4H] = [B,S,H] @ [H,4H]
- FFN down: [B,S,H] = [B,S,4H] @ [4H,H]
- 可共享同一 Cube pipeline 实现

### Gemm
- 直接参数化 M, N, K
- 可附加 bias/epilogue
