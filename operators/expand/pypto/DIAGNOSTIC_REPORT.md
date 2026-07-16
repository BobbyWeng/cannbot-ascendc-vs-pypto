# Expand PyPTO Diagnostic Report

## Status: FIXED

### Issue History
1. **Empty src/ directory** — No implementation file existed
2. **`pypto.op.List` error** — Used type annotation as constructor; fixed with plain `[1, 384]`
3. **"Only allow to expand one axis"** — expand_clone target shape mismatch for 2D; fixed with per-row 1D expand

### Current Implementation
- JIT kernel: `expand_row(x: [1] FP16, y: [384] FP16)`
- Wrapper: reshape input to [rows, 1], squeeze to 1D per-row, expand [1]→[384] per row
- Correctness: B=1 bitwise PASS (max_diff=0.0)

### Remaining
- Per-row dispatch triggers 256 JIT → kernel invocations per batch
- JIT caching avoids recompilation but Python loop overhead remains
- Ascend C still host fallback (no device-side kernel)
