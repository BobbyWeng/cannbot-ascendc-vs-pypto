# Cube API Audit for Ascend C MatMul

## Availability

| API | Status | Notes |
|-----|--------|-------|
| `MatmulImpl` (high-level) | Available | `#include "adv_api/matmul/matmul.h"` in CANN 9.0.0 |
| `MatmulType<GM, ND, half, false>` | Available | ND format FP16 input |
| `TCubeTiling` | Available | `optiling::TCubeTiling` with all tiling fields |
| `IterateAll(GM tensor)` | Available | Single-call Matrix Multiply |
| `SetTensorA/SetTensorB` | Available | GlobalTensor input |

## Data Format Requirements

| Parameter | Required | Our Setting |
|-----------|----------|-------------|
| A layout | ND | ND (RowMajor) |
| B layout | ND (RowMajor) | ND (RowMajor) for outer product; NZ for cube-native |
| C layout | ND | ND |
| M alignment | 16 (fractal) | 256 ✓ |
| N alignment | 16 (fractal) | 32 ✓ |
| K alignment | 16 (fractal) | 256 ✓ |

## L0/L1 Constraints

- L1 can hold ~128KB-256KB per core
- L0A/L0B fixed size per Cube iteration
- For ND format, Cube MMAD internally handles ND→NZ conversion
- Accumulation dtype: FP16 (half) input → FP16 accumulator for `CFG_NORM`

## Workspace

- Not required for basic MatMul (non-batched, non-partial)
- Workspace needed for large batch/partial output modes

## SoC Support

- Tested with dav-2201 (Ascend 910B)
- Cube MMAD available on all dav-200 series+
