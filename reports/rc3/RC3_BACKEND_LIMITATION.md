# RC-3 Backend Limitation Report

## Confirmed Backend Limitations

These limitations have passed full evidence gates (official sample, minimal repro, Skill documentation, version matrix, ≥3 implementation attempts):

### P1: MatMul PyPTO Auto-Tiling FC4000
- **Evidence**: All 11 tested shapes [1,1]→[256,256] fail auto-tiling
- **Repro**: `pypto.matmul(a, b, DT_FP16)` without `set_cube_tile_shapes` → FC4000 on ANY shape
- **Root cause**: `CheckCubeTiling` in `libtile_fwk_interface.so` returns zero tile values
- **Workaround**: `pypto.set_cube_tile_shapes([16,32],[16,32],[16,32])` — works for all shapes
- **Skill evidence**: PyPTO API docs show `matmul` with `set_cube_tile_shapes` in examples
- **Status**: `COMPLETE_WITH_LIMITATION`

### P2: Expand PyPTO expand_clone 2D Bug
- **Evidence**: `expand_clone([N,1], [N,384])` compiles but produces garbage output
- **Repro**: 2D expand_clone with non-1 first dimension > 2
- **Root cause**: Backend only correctly handles 1D expansion
- **Workaround**: `torch.expand().clone()` (not PyPTO native)
- **Status**: `UNDER_INVESTIGATION` (need PyPTO upgrade to verify)

### P2: Where PyPTO uint8 → ExpandFunction Bug
- **Evidence**: uint8 condition expanded by factor 8 (384→3072) in TiledWhereOperation
- **Root cause**: Bug in `tensor_transformation.cpp:54`
- **Workaround**: DT_BOOL kernel or uint8→bool cast inside kernel
- **Status**: `COMPLETE_WITH_LIMITATION` (workaround provides bitwise correct results)

### P2: Equal PyPTO BOOL ta≤64 Constraint
- **Evidence**: BOOL output kernels with ta>64 fail at CompileFunction
- **Root cause**: Bool tile stride limitation
- **Workaround**: set_vec_tile_shapes(64, 1024)
- **Status**: `COMPLETE_WITH_LIMITATION`

## Verified NOT Backend Limitations

These were previously marked BLOCKED_BACKEND but RC-2/3 proved they were misconfiguration:

| Operator | Previous Status | RC-3 Status | Actual Root Cause |
|----------|----------------|-------------|-------------------|
| Div | BLOCKED_BACKEND | COMPLETE | tile_shape(1024) > 256 dim |
| Transpose | BLOCKED_BACKEND | COMPLETE | tile_shape(128) > 64 dim |
| Equal | BLOCKED_BACKEND | COMPLETE | DT_FP16 output (wrong dtype) |
