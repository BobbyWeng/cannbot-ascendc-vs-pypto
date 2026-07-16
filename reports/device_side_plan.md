# Device-side Implementation Plan

## Current State Analysis

### Expand (host fallback: host pre-expand + Add/Sub identity)
**Current kernel**: `expand_kernel.asc` does identity copy (Add+Sub). Host pre-expands [B,256,1]→[B,256,384] on CPU.

**Real issue**: The code never tried to implement device-side expand. It fell back because "Duplicate 3-arg API exists but hits template resolution issues" (from diagnostic report).

**API research** (`ascendc-api-best-practices`):
- `AscendC::Duplicate<half>(dstLocal, scalarValue, count)` — copies scalar to N elements (confirmed working in `asc-devkit/examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/duplicate/duplicate.asc`)
- Duplicate also has `Duplicate(LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)` variant — copies first N elements of src repeatedly. However this may be 950-only.

**Plan**: 
- Load [1] input for each row
- Read value via `GetValue` (for scalar access of 1 element)
- `Duplicate` scalar to fill [384] output row  
- Process [B,256,1] → duplicate each row to [384], output [B,256,384]

### Transpose (host fallback: host transpose + identity kernel)
**Current kernel**: `transpose_kernel.asc` uses `GetValue/SetValue` element-by-element for transpose (API BLACKLIST!). Host pre-transposes on CPU.

**Real issue**: The kernel used `GetValue/SetValue` for tile transpose, which is extremely slow and blacklisted.

**API research**:
- **Basic Transpose**: `AscendC::Transpose<half>(dstLocal, srcLocal)` — 16x16 matrix transpose only
- **Advanced Transpose-96** `TransposeType::TRANSPOSE_ND2ND_021` — Scene 13: 2D transpose [H,W]→[W,H] or 3D last-2-dims transpose [N,H,W]→[N,W,H]
  - Requires `ConfusionTransposeTiling` and optional `sharedTmpBuffer`
  - Requires H,W multiples of 16 ✓ (256,384 both multiples of 16)
  - **Not supported on dav-2201** (constraint says Scene 13-16 is 950 only)
- **Advanced Transpose-96** `TransposeType::TRANSPOSE_ND2ND_ONLY` — Scene 7: 2D transpose
  - Requires H,W multiples of 16 ✓
  - **Supported on dav-2201 (A3/A2)** ✓
  - Uses `TransDataTo5HD` internally for the transpose

**Plan**:
- Use `Transpose-96` Scene 7 (`TRANSPOSE_ND2ND_ONLY`) for 2D transpose of each [H,W] per batch
- Tile [B,256,384] into per-batch [256,384] 2D transpose
- Need `ConfusionTransposeTiling` from host-side tiling computation
- Need the output shape [384,256]

### ReduceSum (host fallback: host FP32 reduce + identity kernel)
**Current kernel**: `reduce_sum_kernel.asc` does identity copy. Host FP32 pre-reduces [B,256,384]→[B,256] on CPU.

**Real issue**: Never implemented device-side reduction.

**API research** (`api-reduce.md`):
- **Level 2 ReduceSum**: `AscendC::ReduceSum<half>(dst, src, sharedTmpBuffer, count)` — per-row reduction, NO alignment requirement
  - Verified in `asc-devkit/examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/reduce_computation/reduce_computation.asc` (Scenario 5)
  - `sharedTmpBuffer` type must match T (half, not uint8_t)
- **Pattern ReduceSum** (`ReduceSum-90.md`): `AscendC::ReduceSum<float, Pattern::Reduce::AR>(dst, src, sharedTmpBuffer, shape, true)` for batch reduce
  - Requires 32-byte aligned columns and float only for A3
  - Our data is half, so Level 2 API is the right choice

**Plan**:
- For each row (B*256 rows of 384 elements):
  - Copy [384] half from GM to UB
  - `ReduceSum<half>(rowResult, srcRow, tmpBuf, 384)`
  - Write 1 element result to GM output
- FP16 accumulation (native to ReduceSum API)

## Implementation Order

1. **ReduceSum** — simplest (direct Level 2 API per row, no tiling complexity)
2. **Expand** — moderate (Duplicate per row, need GetValue for 1-element)
3. **Transpose** — most complex (Scene 7 Advanced Transpose with tiling)

## Key Skills References

| API | Skill Document |
|-----|---------------|
| Duplicate | ascendc-api-best-practices → api-arithmetic.md |
| ReduceSum (Level 2) | ascendc-api-best-practices → api-reduce.md |
| ReduceSum (Pattern) | ascendc-api-best-practices → api-reduce-pattern.md |
| Transpose (Basic) | ascendc-api-best-practices → api-transpose.md |
| Transpose (Advanced-96) | ascendc-docs-search → find Transpose-96.md |
| DataCopyPad | ascendc-api-best-practices → api-datacopy.md |
| Tiling Design | ascendc-tiling-design → conversion/patterns.md (small-channel transpose) |
