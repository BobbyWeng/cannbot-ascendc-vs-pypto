# Key Blocker Repair Summary — layout_reduce_v1

## Overview

This report documents the diagnosis and repair of Expand, Transpose, and ReduceSum PyPTO blockers. All three operators were previously marked `COMPLETE_WITH_LIMITATION` based on incorrect assumptions.

## Root Causes Found

### 1. ReduceSum (repaired)
| Issue | Detail |
|-------|--------|
| Original | `DT_FLOAT16` not found — AttributeError at import |
| Root cause | PyPTO dtype constant is `DT_FP16`, not `DT_FLOAT16` |
| Original | `pypto.op.reduce_sum` not found — AttributeError |
| Root cause | PyPTO API is `pypto.op.sum(input, dim, keepdim)`, not `reduce_sum` |
| Status | **FIXED**. B=1 correctness PASS for all shapes |

### 2. Expand (repaired)
| Issue | Detail |
|-------|--------|
| Original | Empty `src/` and `tests/` directories — no impl file existed |
| Root cause | Orchestrator never completed Stage 5 (implementation) |
| JIT error | `pypto.op.List([1, 384])` → `Type List cannot be instantiated; use list() instead` |
| Fix | Use plain Python list: `[1, 384]` |
| Backend error | `Only allow to expand one axis` — input dim0 != target dim0 |
| Fix | Use 1D per-row expand: `expand_clone(x, [384])` with x shape `[1]` |
| Status | **FIXED**. B=1 correctness PASS (bitwise, max_diff=0.0). Per-row dispatch needed |

### 3. Transpose (repaired)
| Issue | Detail |
|-------|--------|
| Original | Empty `src/` and `tests/` directories — no impl file existed |
| Root cause | Orchestrator never completed Stage 5 |
| JIT error | `JIT cannot get source code` — function defined in wrong scope |
| Fix | Define JIT function in `.py` file, import via `sys.path.insert` |
| Backend error | `Run pass failed` for [256,384] — CompileFunction limit |
| Finding | Transpose works for small tensors [16,32] up to ~32 rows |
| Limitation | PyPTO backend `transpose` hits CompileFunction pass failure for large tensors |
| Status | **PARTIAL**. Small tensors PASS. Production shape [1,256,384] blocked at backend pass |

## PyPTO JIT Pattern (verified working)

All 5 successful operators + 3 repaired operators share:
- `@pypto.frontend.jit` at top level of `src/{op}_impl.py`
- Import via `sys.path.insert(0, '..', 'src')` + `from {op}_impl import {wrapper}`
- Wrapper function (pure Python, not JIT-decorated)
- Tensor reshape to 2D before kernel call
- `y.move(pypto.op.{api}(x))` pattern

## Status Changes

| Operator | Before | After |
|----------|--------|-------|
| Expand | COMPLETE_WITH_LIMITATION (host fallback) | INCOMPLETE (PyPTO works, need device-side Ascend C) |
| Transpose | COMPLETE_WITH_LIMITATION (host fallback) | INCOMPLETE (PyPTO small PASS, large COMPILE_FAIL) |
| ReduceSum | COMPLETE_WITH_LIMITATION (host fallback) | INCOMPLETE (PyPTO PASS, need device-side Ascend C) |
