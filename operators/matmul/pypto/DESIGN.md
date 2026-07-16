# MatMul PyPTO Design

## API Mapping
- `pypto.matmul(A, B)` → `torch.matmul(A, B)`
- Shape: [B,12,256,256] @ [B,12,256,32] → [B,12,256,32]

## Data Flow
Input A, B → pypto.frontend.jit → pypto.matmul → Output Y

## Tiling Strategy
PyPTO handles tiling internally via the framework's default tiling mechanism.
