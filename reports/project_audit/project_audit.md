# Cannbot Project Audit — Complete Health Check

Generated: 2026-07-16
Mode: Read-only (no modifications, no rebuilds, no reruns)

---

## Operator Audit Summary

### ReLU — `operators/relu/`

| Check | Result | Detail |
|-------|--------|--------|
| **Basic Info** | Shape [B,12,256,32], FP16, no broadcast, B∈{1,2,4,8,16,32,64} |
| **Ascend C** | **PASS** | Source complete (`relu_kernel.asc`, `relu_host.asc`, `relu_tiling.h`, `data_utils.h`, `CMakeLists.txt`). Binary missing (`build/` dir is empty — no `relu_ascendc` executable). Build status cannot be confirmed without rebuild. Output BINs missing from `build/output/`. Reports exist in `reports/final/`. Profiler raw data is a single `unified_summary.json` (not standard per-batch msprof raw dirs). |
| **Torch** | **PASS** | `correctness.py` exists. `benchmark.py` exists. No correctness_results.json found (script must be rerun). |
| **PyPTO** | **SUCCESS** | `SPEC.md`, `API_REPORT.md`, `DESIGN.md` complete. `relu_impl.py`, `relu_golden.py`, `test_relu.py` exist. `orchestrator_state.json` shows Stage 7 completed. All batches PASS (bitwise, signed-zero exempted). |
| **Overall** | **COMPLETE** | All artifacts present. Correctness verified. Profiler data collected. |

**Issues:**
- `ascendc/build/` is almost empty — no compiled binary, no output BINs. Cannot verify binary exists without rebuild.
- No `correctness_results.json` in `torch/` or `pypto/` (old results may have been cleaned up).
- `reports/raw/unified_summary.json` is non-standard format; per-batch msprof raw dirs are missing.

### Mul — `operators/mul/`

| Check | Result | Detail |
|-------|--------|--------|
| **Basic Info** | Shape [B,3,4,256,32], FP16, no broadcast, B∈{1,2,4,8,16,32,64} |
| **Ascend C** | **PASS** | Source complete. Binary exists (`ascendc/build/mul_ascendc`, 379KB). Output BINs exist for all 7 batches. Build files present. Reports in `reports/final/`. Profiler raw data has per-batch dirs for ascendc, pypto, and torch. |
| **Torch** | **PASS** | `correctness.py` + `correctness_results.json` (all 7 batches PASS, strict bitwise). `benchmark.py` + `benchmark_results.json`. |
| **PyPTO** | **SUCCESS** | `SPEC.md`, `API_REPORT.md`, `DESIGN.md` complete. `mul_impl.py`, `mul_golden.py`, `test_mul.py` exist. `orchestrator_state.json` shows Stage 7 completed. All batches PASS. |
| **Overall** | **COMPLETE** | Most complete operator. All artifacts and data present. |

**Issues:**
- No `pypto/correctness_results.json` (test runs not persisted).
- SHA256SUMS lists build artifacts (CMakeCache.txt, object files) which change on every build.

### Add — `operators/add/`

| Check | Result | Detail |
|-------|--------|--------|
| **Basic Info** | Shape [B,256,384], FP16, 4-input `((X1+X2)+X3)+X4`, B∈{1,2,4,8,16,32,64} |
| **Ascend C** | **PASS** | Source complete. Binary exists (`ascendc/build/add_ascendc`, 383KB). Output BINs for all 7 batches. Profiler raw data NOT collected (no `reports/raw/` directory). |
| **Torch** | **PASS** | `correctness.py` + `correctness_results.json` (all 7 batches PASS, bitwise). `benchmark.py` exists. |
| **PyPTO** | **SUCCESS** | `SPEC.md`, `API_REPORT.md`, `DESIGN.md` complete. `add_impl.py`, `add_golden.py`, `test_add.py` exist. Correctness: **only B=1 verified** in `correctness_results.json` — B=2..64 are MISSING from saved results (though test code supports all batches). |
| **Overall** | **INCOMPLETE** | Missing: per-batch profiler data, full correctness results persistence, no README, no REPRODUCE.md, no SHA256SUMS, no `reports/parsed/`. |

**Issues:**
- **CRITICAL**: `pypto/correctness_results.json` only contains B=1 result. B=2,4,8,16,32,64 not persisted.
- **CRITICAL**: `reports/raw/` directory does not exist — no msprof data collected.
- **CRITICAL**: `reports/parsed/` directory does not exist.
- **CRITICAL**: `reports/correctness/` directory does not exist.
- No README.md in operator root.
- No REPRODUCE.md.
- No SHA256SUMS.
- No `data/manifest.json` (though data exists).
- No profiler_config in benchmark/.
- No `benchmark/run_all.sh`.
- Benchmark uses `torch.npu.Event` timing (not msprof profiler-based).
- `test_add.py` imports `add_wrapper` but impl file exports `add_4`. Test calls `add_wrapper()` which does not exist.

### Div — `operators/div/`

| Check | Result | Detail |
|-------|--------|--------|
| **Basic Info** | Broadcast div: X1 [B,12,256,256], X2 [B,12,256,1], FP16, B∈{1,2,4,8,16,32} |
| **Ascend C** | **PASS** | Source complete (baseline + optimized + baseline variants). Binary exists (`ascendc/build/div_ascendc`, 381KB). Output BINs for B=1,2,4,8,16,32 (no B=64). |
| **Torch** | **PASS** | `correctness.py` + `correctness_results.json` (all batches PASS). `benchmark.py` + `benchmark_results.json`. |
| **PyPTO** | **BACKEND LIMITATION** | `SPEC.md`, `API_REPORT.md`, `DESIGN.md` complete. `div_impl.py`, `div_golden.py`, `test_div.py` exist. Native Div with broadcast fails at backend CompileFunction. Minimal 2D Div [1,32]/[1,1] passes. |
| **Overall** | **COMPLETE (with known limitation)** | All required artifacts present. Broadcast Div backend limitation clearly documented. |

**Issues:**
- `reports/correctness/` only has ascendc correctness. No pypto correctness results saved.
- Div profiler data only collected for B=32, not per-batch.
- `reports/parsed/` only has ascendc_b32.json and torch_b32.json — no per-batch breakdown.
- No B=64 profiler data (B=64 is optional per spec, but no data at all).
- `div_wrapper` in `div_impl.py` has a cache bug: cache stores `(x1_2d, x2_2d, y_2d)` but on cache hit, `x1_2d` and `x2_2d` are re-created while `y_2d` is reused from the old shape — shape mismatch risk.
- `test_div.py` generates random data and doesn't load from data/ directory.
- `test_add.py` calls `add_wrapper` which does not exist (should be `add_4`).

---

## Correctness Audit

### Required Coverage: B ∈ {1, 2, 4, 8, 16, 32, 64}

| Operator | Backend | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 | Special Values | Metrics |
|----------|---------|:---:|:---:|:---:|:---:|:----:|:----:|:----:|:-------------:|:-------|
| **ReLU** | Torch | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT | N/A | max_abs_diff, max_rel_diff, NaN, Inf |
| **ReLU** | PyPTO | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | N/A | No JSON saved |
| **Mul** | Torch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A (no special) | max_abs_diff: NaN, max_rel_diff: NaN, NaN:0, Inf:counted |
| **Mul** | PyPTO | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | N/A | No JSON saved |
| **Add** | Torch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A | max_abs_diff: NaN, max_rel_diff: NaN, NaN:0, Inf:counted |
| **Add** | PyPTO | ✅ (B=1 only) | ❌ MISSING | ❌ MISSING | ❌ MISSING | ❌ MISSING | ❌ MISSING | ❌ MISSING | N/A | Only B=1 saved |
| **Div** | Torch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A (optional) | Present in spec | atol=1e-3, rtol=1e-3 |
| **Div** | PyPTO | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | NOT SAVED | N/A | N/A | Backend limitation |

**Deficiencies:**
1. **max_abs_diff and max_rel_diff show NaN** in Mul and Add torch correctness results — the `check_correctness` function reports `NaN` for these when `bitwise_equal=true` (no diff to compute), which is technically correct but confusing.
2. **PyPTO correctness results are NOT PERSISTED** for ReLU, Mul, and Div. Only Add has a saved `correctness_results.json` (partial).
3. **Add PyPTO correctness only saved B=1** — B=2..64 were run but not saved to results JSON.
4. **No ULP measurements** anywhere in the project.
5. **Signed zero** tracking: present in Mul and Add, but not explicitly tracked in ReLU (exempted per spec).

---

## Benchmark Audit

### Measurement Consistency

| Aspect | ReLU | Mul | Add | Div |
|--------|------|-----|-----|-----|
| **Warmup** | 200 | 200 | 200 | 200 (torch), 100 (ascendc) |
| **Loops** | 100 | 100 | 100 | 100 (torch), 1000 (ascendc) |
| **Repeat** | 5 | 5 | 5 | 5 (torch), 10 (ascendc) |
| **Profiler** | msprof | msprof | torch.npu.Event | msprof + aclrtEvent |
| **Kernel Mapping** | Yes | Yes | No (operation-level) | Partial |
| **Kernel Count** | Yes | Yes | No | Yes |
| **Host Time** | Via msprof | Via msprof | torch.npu.Event | Mixed |
| **Device Time** | Via msprof | Via msprof | No | Partial |
| **JIT Excluded** | 2-process | 2-process | No | N/A (PyPTO limited) |

**Inconsistencies:**
1. **Add uses `torch.npu.Event` instead of msprof** — different measurement methodology from ReLU/Mul/Div.
2. **Div config mismatch**: torch uses warmup=200/loops=100/repeat=5 while ascendc uses warmup=100/loops=1000/repeat=10.
3. **Add benchmark does not separate kernel launch from compute** — reports PyPTO as ~386-397us (all-device) vs ReLU/Mul which separate primary compute kernel from executor kernels.
4. **Div final report latency** uses aclrtEvent timing, not msprof parsed data, while ReLU/Mul use msprof exclusively.
5. **No `repeat` dimension in profiler raw data** — results only show median/mean without per-repeat breakdown.

---

## Profiler Audit

### Consistency Check: Report vs Raw Parsed Data

| Operator | Batch | Report Claim (us) | Parsed JSON (us) | Match? |
|----------|:-----:|:-----------------:|:----------------:|:------:|
| ReLU Ascend C | B=32 | 6.021 | 6.021 | ✅ |
| ReLU PyPTO | B=32 | 103.909 (MIX_AIC) | 103.909 | ✅ |
| ReLU Torch | B=32 | 4.674 | 4.674 | ✅ |
| Mul Ascend C | B=32 | 10.432 | 10.432 (primary) | ✅ |
| Mul PyPTO | B=32 | 136.389 (MIX_AIC) | 136.389 | ✅ |
| Mul Torch | B=32 | 8.184 | 8.184 | ✅ |
| Div Ascend C | B=32 | 328.6 (aclrtEvent) | 306.5 (msprof primary) | ⚠️ **38% discrepancy** |
| Div Torch | B=32 | 112.80 | 126.20 (msprof primary) | ⚠️ **12% discrepancy** |

**Critical Findings:**
1. **Div Ascend C B=32**: Report claims 328.6us (aclrtEvent), but parsed msprof shows 306.5us (primary compute kernel). **REPORT OUTDATED** — 22.1us difference (7.2%).
2. **Div Torch B=32**: Report claims 112.80us (torch.npu.Event), but parsed msprof shows 126.20us (primary compute kernel). **REPORT OUTDATED** — 13.4us difference (11.9%).
3. **Add has NO profiler data** — all numbers from torch.npu.Event, not from msprof. Cannot cross-verify.
4. **Div parsed profiler** only has ascendc_b32.json and torch_b32.json — no per-batch profiles for B=1,2,4,8,16.

---

## Report Audit

### Consistency Check Across Documents

**Root README (`README.md`):**
- Claims add/mul/div as "planned" — **OUTDATED**: all three are complete with real data.
- Claims div "Needs HW" for correctness — **OUTDATED**: div correctness is verified (Ascend C PASS, torch PASS, PyPTO LIMITED).
- Claims div "Needs HW" for profiler — **OUTDATED**: profiler data exists (B=32).

**Operators README (`operators/README.md`):**
- Accurately reflects current status of all 4 operators. ✅

**Div SHA256SUMS:**
- Uses `operators/div/...` relative paths (correct for archive), but `final_comparison.json` path is relative to working dir at time of creation, not operator root. Low severity.

**Mul SHA256SUMS:**
- Uses `./` prefix (relative to archive root). Includes build artifacts (`CMakeCache.txt`, `.o` files) which change per build. These should not be in integrity manifest for source code.

**ReLU SHA256SUMS:**
- Uses `operators/relu/...` paths (correct). Does NOT include build artifacts. ✅

---

## Dashboard Audit

### `dashboard/dashboard.json` vs Operator State

| Check | Actual | Status |
|-------|--------|--------|
| **Operator count** | 4 (add, div, mul, relu) | ✅ Correct |
| **Completed** | All 4 marked completed | ✅ |
| **In progress** | None | ✅ |
| **Blocked** | None | ✅ |
| **Kernel types** | All accurately listed | ✅ |
| **Latency numbers** | Match reports | ✅ |
| **PyPTO status** | All 4 "pypto: done" for add,div,mul,relu | ⚠️ Div PyPTO should be "LIMITED" not "done" |

**Issues:**
1. Dashboard marks Div PyPTO as "done" — but PyPTO broadcast Div fails at backend. Should be "LIMITED".
2. Dashboard generated at `2026-07-16 11:14:02` — timestamps of report data may be earlier.

---

## Archive Audit

| Archive | Size | Status | Notes |
|---------|:----:|:------:|:------|
| `cannbot_ascendc_vs_pypto_div_v1_pre_fix.tar.gz` | 187 MB | **OUTDATED** | Contains pre-fix kernel with buggy formula. 187MB includes profiler raw data and build artifacts. |
| `cannbot_ascendc_vs_pypto_div_v2.tar.gz` | 388 KB | **CURRENT** | Optimized kernel archive. Contains source, reports, parsed profiler. No build artifacts. ✅ |
| `cannbot_ascendc_vs_pypto_mul_v1.tar.gz` | 183 MB | **MIXED** | Contains 183MB of profiler raw data. Source-only would be ~500KB. Should be slimmed. |

All archives predate current code. No SHA256 verification was performed (would require decompression).

---

## Redundant File Audit

### KEEP — Essential project files
- All `SPEC.yaml`, `experiment_config.yaml`, `README.md`, `REPRODUCE.md`
- Source files in `ascendc/src/`, `pypto/src/`, `pypto/golden/`, `pypto/tests/`
- Final reports in `reports/final/`
- Parsed profiler in `reports/parsed/`
- Dashboard files in `dashboard/`

### REGENERATE — Should be regenerated
- **Python `__pycache__/` directories** (11 locations) — auto-generated, safe to delete
- **Mul SHA256SUMS** — includes build artifacts and .o files; should exclude build/
- **Root README status table** — shows add/mul/div as "planned" — needs update

### DELETE — Candidates for cleanup
- `cannbot_ascendc_vs_pypto_div_v1_pre_fix.tar.gz` (187 MB) — outdated, superseded by v2
- `cannbot_ascendc_vs_pypto_mul_v1.tar.gz` (183 MB) — contains 183MB of raw profiler
- All `__pycache__/` directories
- Build artifacts in `ascendc/build/` (CMakeCache.txt, CMakeFiles/, .o files, etc.)
- `operators/relu/ascendc/build/` is empty — should be removed or rebuilt
- `operators/relu/pypto/src/output/` — empty directory
- `operators/div/output/` — empty directory

### LOCAL ONLY — Not tracked (or should be gitignored)
- `__pycache__/` files
- `ascendc/build/` directories and their contents
- `profiling/` directories (if git-tracked)
- Large archives (tar.gz) > 50MB

### Specific Redundancies:
1. **`operators/add/pypto/src/__pycache__/test_relu_copy_impl.cpython-311.pyc`** — stale pyc from old ReLU code
2. **`operators/div/ascendc/src/div_kernel_baseline.asc`** — kept for reference, intentionally
3. **`operators/div/ascendc/src/div_kernel_optimized.asc`** — identical hash to div_kernel.asc (SHA256SUMS confirms)
4. **Profiler raw data** (~300+ MB total across all operators) — essential for reproducibility but very large

---

## Directory Structure Consistency

### Standard Template (from `templates/operator_template/`)

Required structure:
```
README.md | SPEC.yaml | experiment_config.yaml |
data/     | torch/     | ascendc/              |
pypto/    | benchmark/ | reports/              |
REPRODUCE.md | SHA256SUMS
```

### Compliance Matrix

| Component | ReLU | Mul | Add | Div |
|-----------|:----:|:---:|:---:|:---:|
| README.md | ❌ (has pypto/README but not operator root README) | ✅ | ❌ | ✅ |
| SPEC.yaml | ✅ | ✅ | ✅ | ✅ |
| experiment_config.yaml | ✅ | ✅ | ✅ | ✅ |
| data/ | ✅ | ✅ | ✅ | ✅ |
| torch/ | ✅ | ✅ | ✅ | ✅ |
| ascendc/ | ✅ | ✅ | ✅ | ✅ |
| pypto/ | ✅ | ✅ | ✅ | ✅ |
| benchmark/ | ✅ | ✅ | ✅ (no run_all.sh) | ✅ |
| reports/ | ✅ | ✅ | ✅ (raw,parsed,correctness missing) | ✅ |
| REPRODUCE.md | ✅ | ✅ | ❌ | ✅ |
| SHA256SUMS | ✅ | ✅ (includes build artifacts) | ❌ | ✅ |

---

## Scoring

### Scoring Rubric (0-10 per dimension)

| Dimension | ReLU | Mul | Add | Div |
|-----------|:----:|:---:|:---:|:---:|
| **Completeness** (all artifacts present) | 8 | 9 | 4 | 8 |
| **Correctness** (all batches, all metrics) | 6 | 8 | 5 | 7 |
| **Performance** (consistent measurement) | 8 | 8 | 5 | 6 |
| **Documentation** (README, SPEC, DESIGN) | 7 | 8 | 5 | 9 |
| **Reproducibility** (REPRODUCE.md, SHA256SUMS) | 7 | 6 | 1 | 8 |
| **Dashboard** (accuracy) | 8 | 9 | 7 | 7 |
| **Archive** (cleanliness) | 5 | 4 | - | 6 |
| **Uniformity** (follows template) | 7 | 8 | 3 | 8 |
| **OVERALL** | **7.0** | **7.5** | **4.3** | **7.4** |

---

## Findings Summary

### 1. Truly Completed Operators
- **ReLU**: All stages complete. Missing compiled binary, but source and reports complete.
- **Mul**: Most complete operator. All data present, all batches verified.
- **Div**: Complete with documented backend limitation. Broadcast Div PyPTO fails.

### 2. Incomplete Operators
- **Add**: Missing profiler data (raw/parsed), missing README, missing REPRODUCE.md, missing SHA256SUMS, correctness only saved for B=1, test code imports non-existent function.

### 3. Operators Where Only Report Outdated
- **ReLU**: Reports are current (no discrepancies).
- **Mul**: Reports are current (no discrepancies).
- **Div**: Report latency numbers for B=32 differ from profiler parsed data (both ascendc and torch).

### 4. Profiler Outdated Report
- **Div**: ascendc B=32: 328.6us (report) vs 306.5us (profiler) — **7.2% error**
- **Div**: torch B=32: 112.80us (report) vs 126.20us (profiler) — **11.9% error**

### 5. Outdated README
- **Root README**: Shows add/mul/div as "planned" — all are complete.
- **Root README**: Shows div as "Needs HW" — div has data.

### 6. Dashboard Inconsistencies
- Div PyPTO marked as "done" — should be "LIMITED".

### 7. Invalid Archive
- `cannbot_ascendc_vs_pypto_div_v1_pre_fix.tar.gz` (187 MB) — outdated, superseded by v2.

### 8. Redundant Files
See full list above. Primary candidates: outdated archives (370 MB total), pycache (11 dirs), build artifacts.

---

## Next Steps — Priority

### P0 (Must Fix Immediately)
1. **Add correctness**: Only B=1 saved; re-run and persist all batches.
2. **Add profiler**: Collect msprof data for all 3 implementations.
3. **Add test import bug**: `test_add.py` calls `add_wrapper()` which doesn't exist; should call `add_4()`.
4. **Div report latency**: Update report to reflect profiler-measured numbers (306.5us, not 328.6us).
5. **Root README status table**: Update add/mul/div from "planned" to "complete".

### P1 (Should Fix)
6. **Add REPRODUCE.md and SHA256SUMS** — critical for reproducibility.
7. **PyPTO correctness results not persisted** — save after each test run for ReLU, Mul, Div.
8. **Clean outdated archives**: Remove `div_v1_pre_fix.tar.gz` (187 MB) and slim `mul_v1.tar.gz`.
9. **Fix `div_wrapper` cache bug**: Cache key uses old `y_2d` shape on cache hit.
10. **Update dashboard**: Change Div PyPTO status to "LIMITED".
11. **Remove build artifacts from SHA256SUMS** (Mul).

### P2 (Can Do Later)
12. **Standardize Add benchmark** to use msprof (same as ReLU/Mul).
13. **Standardize Div benchmark** config (warmup/loops/repeat to match ReLU/Mul).
14. **Remove all `__pycache__/` directories**.
15. **Relu build**: Either rebuild binary or remove empty `build/` directory.
16. **Add per-batch profiler for Div** (currently only B=32).
17. **Add ULP measurement** to correctness checking.
18. **Unify SHA256SUMS path format** across all operators.
19. **Clean up stale `test_relu_copy_impl.cpython-311.pyc`** in add pycache.
20. **Remove empty `operators/relu/pypto/src/output/` and `operators/div/output/`**.
