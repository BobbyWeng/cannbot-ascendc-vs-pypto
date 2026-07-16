# PyPTO MatMul API Report

## API Availability

| API | Status | Notes |
|-----|--------|-------|
| `pypto.matmul` | Available | Top-level function |
| `pypto.frontend.jit` | Available | Can compile lambdas with `.matmul()` |
| `.matmul()` tensor method | Available | Inside JIT-compiled functions |

## API Mapping

| PyPTO API | PyTorch Equivalent | Status |
|-----------|-------------------|--------|
| `pypto.matmul(A, B)` | `torch.matmul(A, B)` | ✓ Available |
| `jit(lambda a,b: a.matmul(b))` | `torch.matmul` | ✓ Available |

## Shape Support

| Shape | Expected | Status |
|-------|----------|--------|
| [1,16,16] @ [1,16,16] | [1,16,16] | P0 gate |
| [1,256,256] @ [1,256,32] | [1,256,32] | P0 gate |
| [12,256,256] @ [12,256,32] | [12,256,32] | P0 |
| [B,12,256,256] @ [B,12,256,32] | [B,12,256,32] | P0 full |

## Constraints

- Input dtype: FP16
- Output dtype: FP16
- Accumulation: FP16 (default), may differ from FP32
