# Div Audit: Current State Assessment

## 1. Source Code Audit

### 1.1 Kernel Source (`ascendc/src/div_kernel.asc`)

**Formula check**: Does kernel compute `X1 / X2` or `1/(X1*X2)`?

Line 88: `AscendC::Div(yLocal[rowOff], x1Local[rowOff], x2Broadcast, segLen);`

**Verdict**: CORRECT — uses native `Div`. No `1/(X1*X2)` residue found.

### 1.2 X1/X2 Address Calculation

Line 22-24:
```
x1Addr = (__gm__ half*)x1;
x2Addr = (__gm__ half*)x2;
yAddr = (__gm__ half*)y;
```

Line 25: `blockOffset = blockIdx * tiling->numPerCore;`

X1 CopyIn (line 56-63):
```
baseOffset = blockOffset + tileIdx * TILE_LENGTH;
x1Gm.SetGlobalBuffer(x1Addr + baseOffset, count);
```
CORRECT — contiguous access from X1.

X2 CopyIn (line 65-72):
```
x2Off = baseOffset / X2_DIVISOR;    // row index in X2 (256-wide rows)
x2Cnt = (count + X2_DIVISOR - 1) / X2_DIVISOR;
x2Gm.SetGlobalBuffer(x2Addr + x2Off, x2Cnt);
```
**ISSUE**: X2 address calculation assumes X2 is contiguous with 256-wide rows, and that X2 has 256 rows per core block (same as X1). This is correct ONLY if `X2_DIVISOR = 256 = number of elements per row in X2`. Since X2 shape is `[B,12,256,1]`, each "row" of X2 is 1 element, but `X2_DIVISOR = 256`. The division `baseOffset / X2_DIVISOR` computes the *row index in X1*, which is the same as the scalar index in X2. **This is correct** because every 256 elements of X1 share the same X2 scalar.

### 1.3 Broadcast Index Calculation

Compute (line 82-88):
```
rows = (count + X2_DIVISOR - 1) / X2_DIVISOR;    // 8192/256 = 32 rows per tile
for (uint32_t r = 0; r < rows; r++) {
    half x2Val = x2Local(r);          // read 1 scalar from the 32 X2 scalars
    AscendC::Duplicate(x2Broadcast, x2Val, X2_DIVISOR);   // broadcast to 256 elements
    rowOff = r * X2_DIVISOR;
    segLen = min(256, remaining);
    AscendC::Div(yLocal[rowOff], x1Local[rowOff], x2Broadcast, segLen);
}
```

**Verdict**: Broadcast logic is correct — each row uses 1 X2 scalar, Duplicate to 256, then Div.

### 1.4 Tile Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| TILE_LENGTH | 8192 | Elements per tile |
| Rows per tile | 32 | 8192/256 = 32 rows of 256 |
| DOUBLE_BUFFER | 2 | 2-buffer queue depth |
| blockDim (default) | 20 | From div_host.asc line 183 |

### 1.5 Buffer Sizing

| Buffer | Size | Count |
|--------|------|-------|
| inQueueX1 | 8192 * 2B = 16 KB | ×2 (double buffer) = 32 KB |
| inQueueX2 | 32 * 2B = 64 B | ×2 (double buffer) = 128 B |
| bufBroadcast | 8192 * 2B = 16 KB | ×1 = 16 KB |
| outQueueY | 8192 * 2B = 16 KB | ×2 (double buffer) = 32 KB |

Total UB: ~80 KB — well within typical 192 KB UB limit.

**ISSUE**: X2 buffer is tiny (128 B). This reflects the per-row Duplicate approach.

### 1.6 Pipeline

- CopyIn: DataCopyPad for X1 (8192 elements) + DataCopyPad for X2 (32 elements)
- Compute: Per-row loop: scalar read → Duplicate → Div
- CopyOut: DataCopyPad for Y (8192 elements)
- Double buffering: ENABLED for all queues
- **Scalar loop**: The `for (uint32_t r = 0; r < rows; r++)` loop is the main bottleneck

## 2. Existing Correctness Verification

### 2.1 What was verified

From `final_comparison.json`:
- Ascend C: "Exact match with FP32→FP16 reference at B=1,2 (all 786432 elements per batch)"
- Torch: "torch.div matches reference; B=1 PASS, B=2 PASS"

### 2.2 Missing coverage

B=4, 8, 16, 32 are **NOT verified** for Ascend C.
B=4, 8, 16, 32 are **SKIPPED** for torch (missing `reference_b*_fp16.bin` files).

**Verdict**: Current correctness data is incomplete. Only B=1,2 verified.

## 3. Profiler Raw Data Audit

### 3.1 Profiling session exists

Directory `profiling/PROF_000001_20260716110804431_00158189IBLAGBEL/` contains:
- `device_0/sqlite/ai_core_op_summary.db` — AI core summary
- `device_0/sqlite/ascend_task.db` — Task trace
- `device_0/sqlite/metric_summary.db` — Metrics
- `host/sqlite/` — Host-side events

### 3.2 Profiler data quality

SQLite databases exist and are non-empty. The profiler session from 2026-07-16 appears genuine.

### 3.3 Reported profiler metrics

From `final_comparison.json` (lines 56-68):
```
kernel_time_us: 375.7
aiv_vec_time_us: 159.7 (42.3%)
aiv_scalar_time_us: 362.7 (96.0%)
aiv_mte2_time_us: 74.8 (19.8%)
aiv_mte3_time_us: 28.9 (7.7%)
```

**ISSUE**: 362.7 µs scalar time + 159.7 µs vector time > 375.7 µs kernel time. This indicates **overlap** — scalar and vector execute concurrently. The 96% scalar ratio means scalar operations occupy 96% of the kernel duration but overlap significantly with vector compute.

### 3.4 What batch was profiled?

The profiler was run on B=32 (per `final_comparison.json` line 58). No per-batch breakdown exists.

### 3.5 Parsed profiler data

The `reports/parsed/` directory is empty — msprof results were never parsed/aggregated. The `benchmark/parse_profiler.py` was never actually executed with correct args.

### 3.6 Reported latency source

The B=1 through B=32 latency numbers (12.8, 25.8, 46.2, 77.1, 164.4, 327.6 µs) come from the `div_ascendc` binary's aclrtEvent timing, **not from the parsed profiler**. These are direct kernel launch timings, not all-device-kernels-per-call.

## 4. Final Report Assessment

### 4.1 Correctness claims

**Claim**: "Exact match with FP32→FP16 reference at B=1,2 (all 786432 elements per batch)"

**Verification needed**: Need to re-run with all batches and full special-value coverage.

### 4.2 Performance data

| Batch | Ascend C (µs) | Torch NPU (µs) | Ratio |
|-------|:------------:|:-------------:|:----:|
| 1 | 12.8 | 14.84 | 0.86× |
| 2 | 25.8 | 14.07 | 1.83× |
| 4 | 46.2 | 18.90 | 2.44× |
| 8 | 77.1 | 31.49 | 2.45× |
| 16 | 164.4 | 57.66 | 2.85× |
| 32 | 327.6 | 112.80 | 2.90× |

Torch data from `benchmark_results.json` — matches.
Ascend C data from `final_comparison.json` — matches reported values.
**No raw latency CSV or per-run logs exist for Ascend C.**

### 4.3 Root cause analysis

**96% scalar overhead claim**: The profiler raw data supports high scalar activity, but:
1. The scalar and vector units overlap on AIV — scalar time is NOT additive to vector time
2. The per-row scalar loop executes 32 × 12 × 256 × B iterations of:
   - Scalar read from X2 buffer
   - Duplicate setup (scalar-to-vector broadcast)
   - Div invocation setup
3. At B=32, total scalar operations = 32 × 12 × 256 × 32 × 32 = 100,663,296 scalar operations

**The real bottleneck**: For every 256-element row, the kernel:
1. Reads 1 FP16 scalar from X2 (inefficient 64-bit DMA read)
2. Calls `Duplicate` to create 256-element broadcast
3. Calls `Div` on 256 elements
4. Repeats 32 times per tile

This is fundamentally 32× more Duplicate+Div invocations than necessary.

## 5. Archive Audit

### 5.1 Archive exists

`cannbot_ascendc_vs_pypto_div_v1.tar.gz` at project root.

### 5.2 What's included

Standard project structure — source, data, reports. Contains:
- Compiled binaries (`div_ascendc`)
- Build artifacts (CMake object files)
- Output BIN files (B=1 through B=32)
- Profiler raw data (single session)
- Final reports (JSON, MD, CSV)

### 5.3 Not included

- Raw per-run timing logs
- Multiple profiler runs (only B=32)
- Per-candidate optimization results
- Correctness results for B=4,8,16,32

## 6. Key Issues Summary

| # | Issue | Severity | Evidence |
|---|-------|----------|----------|
| 1 | B=4,8,16,32 correctness NOT verified | HIGH | correctness_results.json shows SKIP for B>=4 |
| 2 | Per-row Duplicate+Div wastes 32× scalar overhead | HIGH | Kernel code analysis + profiler 96% scalar |
| 3 | Reports/parsed/ directory empty | MEDIUM | Directory exists but no JSON files |
| 4 | Profiler only on B=32, no per-batch breakdown | MEDIUM | Single PROF_* directory |
| 5 | No recip+mul or batch broadcast alternatives explored | HIGH | Only one kernel variant exists |
| 6 | SHA256SUMS uses operators/div/ paths (archive-relative) | LOW | Expected for archive manifest |
