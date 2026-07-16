# Remaining Issues After Repair

## P0 — Must Fix (requires NPU hardware)

### 1. Add Profiler Data Collection
- **What**: Run msprof for torch 4-input add, Ascend C fused add, PyPTO chained adds
- **Why**: Current `comparison_report.md` uses torch.npu.Event timing, not msprof
- **How**: `bash operators/add/benchmark/run_all.sh` on Ascend 910B
- **Files affected**: `operators/add/reports/raw/`, `reports/parsed/`, `reports/final/`

### 2. Add PyPTO Correctness Re-run
- **What**: Execute `python3 operators/add/pypto/correctness.py --batch 1,2,4,8,16,32,64`
- **Why**: Current correctness_results.json only has B=1 verified; B=2..64 need HW
- **How**: On Ascend 910B with PyPTO installed

### 3. ReLU Ascend C Rebuild
- **What**: Rebuild `operators/relu/ascendc/build/relu_ascendc`
- **Why**: build/ directory is empty
- **How**: `cmake -S operators/relu/ascendc -B operators/relu/ascendc/build && cmake --build operators/relu/ascendc/build -j`

---

## P1 — Should Fix

### 4. Mul Archive Slim
- **What**: Create `cannbot_ascendc_vs_pypto_mul_v1_slim.tar.gz` (~500 KB) without raw profiler
- **Why**: Current archive is 183 MB, mostly raw profiler data
- **How**: Exclude `reports/raw/` and `profiling/` from archive

### 5. Div Per-Batch Profiler
- **What**: Collect msprof for Div B=1,2,4,8,16 (currently only B=32)
- **Why**: Incomplete profiler coverage for full comparison

### 6. PyPTO Div Backend Investigation
- **What**: Debug `CompileFunction` failure for broadcast Div
- **Why**: Only minimal 2D Div works; broadcast fails
- **Where**: `libtile_fwk_interface.so` in backend

### 7. Add Final Report Update
- **What**: Update `operators/add/reports/final/comparison_report.md` with msprof data
- **Why**: Currently uses torch.npu.Event; needs device-kernel-level comparison

---

## P2 — Can Do Later

### 8. ULP Measurement
- **What**: Add ULP (Units in Last Place) to `common/correctness/correctness.py`
- **Why**: More precise correctness metric for FP16

### 9. SHA256SUMS Path Standardization
- **What**: Unify all SHA256SUMS to use same relative path convention
- **Why**: Mul uses `./` prefix, Div uses `operators/div/` prefix

### 10. LOCAL_ARTIFACTS.md
- **What**: Document what large local-only files exist
- **Why**: Make it clear what's excluded from Git

### 11. Mul Slim Archive
- **What**: Generate slim archive (same task as P1 #4, but lower urgency)
- **Why**: Current archive size is acceptable for local use
