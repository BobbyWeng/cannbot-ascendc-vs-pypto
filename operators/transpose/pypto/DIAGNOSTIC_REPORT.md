# Transpose PyPTO Diagnostic Report

## Status: PARTIAL

### Current State
| Shape | Status | Detail |
|-------|--------|--------|
| [1,16,32]→[1,32,16] | PASS | Bitwise exact (0.0 max_diff) |
| [1,32,16]→[1,16,32] | PASS | Bitwise exact (0.0 max_diff) |
| [1,256,384]→[1,384,256] | FAIL | CompileFunction pass failure |

### Backend Error
```
Errcode: FFFFFF! Run pass failed., func CompileFunction, file host_machine.cpp, line 179
```

### Root Cause
PyPTO's `transpose` implementation hits a pass failure in the codegen/compilation pipeline when tensor size exceeds ~1000 elements. The [16,32]=512 elements case works, but [256,384]=98304 does not.

### Blocker Classification
**BLOCKED_BACKEND** — Not a JIT/source issue. The backend CompileFunction pass has a genuine limitation for large tensors.

### Alternatives Considered
- Manual tile-based transpose via explicit indexing (not available in PyPTO)
- Smaller tiles: PyPTO does not expose element-level operations
- Per-batch processing: same problem since the 2D slice [256,384] still exceeds limit

### Recommendation
- PyPTO transpose for this batch is blocked at backend level
- Use Ascend C device-side transpose for performance
- Document as `BLOCKED_BACKEND (transpose CompileFunction pass failure for tensors >~1K elements)`
