# Transpose PyPTO Diagnostic Report

## Status: FIXED

### Previous Diagnosis (Incorrect)
Previously believed to be **BLOCKED_BACKEND** — a fundamental CompileFunction limitation for tensors >~1000 elements. The real error was:

```
Errcode: FFFFFF! Run pass failed., func CompileFunction, file host_machine.cpp, line 179
```

### Actual Root Cause
**Tile shape mismatch.** The original implementation used `pypto.set_vec_tile_shapes(128, 1024)`, which causes CompileFunction to fail for tensors whose dimensions don't align well with this tile shape. The scalar tile of 1024 creates too large a per-tile memory footprint.

### Fix
Changed `pypto.set_vec_tile_shapes(128, 1024)` → `pypto.set_vec_tile_shapes(64, 256)` in `transpose_impl.py`.

### Verified Results

| Shape | Elements | Status | Max Diff |
|-------|----------|--------|----------|
| [1,16,32] | 512 | PASS | 0.0 (bitwise) |
| [1,32,16] | 512 | PASS | 0.0 (bitwise) |
| [1,256,384] | 98304 | PASS | 0.0 (bitwise) |
| [1,384,256] | 98304 | PASS | 0.0 (bitwise) |
| [2,16,32] | 1024 | PASS | 0.0 (bitwise) |
| [4,32,64] | 8192 | PASS | 0.0 (bitwise) |
| [2,256,384] | 196608 | PASS | 0.0 (bitwise) |
| [1,1024,1024] | 1048576 | PASS | 0.0 (bitwise) |
| [1,2048,2048] | 4194304 | PASS | 0.0 (bitwise) |

### Conclusion
**Not BLOCKED_BACKEND** — the transpose operation works correctly for all tested sizes up to 4M elements when configured with an appropriate tile shape `(64, 256)`. The issue was purely a configuration problem.
