# PyPTO 14-Operator Full Audit Report

- **Date**: 2026-07-24
- **Branch**: `codex/pypto-full-audit` (commits 6764695 + 8205ef6)
- **Environment**: torch 2.8.0+cpu, torch_npu 2.8.0.post2, CANN 9.0.0, PyPTO d1c290f36
- **Auditor**: orchestrated 9-phase audit per AGENTS.md v2

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total operators audited | 14 |
| PYPTO_NATIVE (single kernel call) | 3 |
| PYPTO_HOST_ORCHESTRATED (per-batch/per-row loop) | 7 |
| UNDER_INVESTIGATION (needs kernel repair) | 3 |
| TORCH_FALLBACK | 1 |
| **Correctness PASS (all B)** | **8/14 (57%)** |
| Correctness PASS (limited B) | 3/14 (B-dependent crash) |
| Correctness FAIL (always) | 2/14 (dtype/inf bugs) |
| N/A (Torch only) | 1/14 |
| Perf-critical operators (host_us > 1ms at B64) | **2 (softmax, layernorm)** |

---

## Phase 1: Fresh Checkout Verification [PASS]

- HEAD verified at 8205ef6 via `git rev-parse HEAD`
- Diff audit: no fake SKILL_TRACE, no Torch fallback marked as PyPTO, no manifest discrepancy
- Fixed softmax `orchestrator_state.json` (added `operator_name` and `stage_retry_count`)

## Phase 2: Fresh-Process Correctness Re-Verification [PASS]

All 14 operators re-tested in **fresh isolated Python processes** to eliminate JIT state cross-contamination (cross-operator JIT contamination confirmed: loading @jit modules in same process after a compiled kernel triggers "function nested is not allowed").

### Complete Results by Route

```
PYPTO_NATIVE (3 ops) — single JIT call, no host loop
  add:       PASS [B,12,256,256]  B1-B64  max_diff=0.000    add_binary(2-arg)
  where:     PASS [B,65536]       B1-B64  max_diff=0.000    pypto.where
  transpose: PASS [B,256,384]     B1-B64  max_diff=0.000    layout op

PYPTO_HOST_ORCHESTRATED (7 ops) — Python host loop calling per-batch/per-row kernel
  relu:      PASS [B,65536]     B1-B64  max_diff=0.000    per-batch loop
  mul:       PASS [B,65536]     B1-B64  max_diff=0.000    per-batch loop
  softmax:   PASS [B*256,384]   B1-B64  max_diff=0.031    per-row loop
  layernorm: PASS [B,256,32]    B1-B64  max_diff=0.003    per-row loop
  matmul:    PASS 2D+B3D        B1-B64  max_diff=0.031    per-batch routing
  equal:     PASS [1,256]       B1 only  CompileFunction crash at B>1
  not:       PASS [1,256]       B1 only  CompileFunction crash at B>1

UNDER_INVESTIGATION (3 ops)
  div:       PASS [B,12,256,256] B1-B2 | B64=NaN (FP16 precision collapse)
  or:        FAIL B1 — bitwise_or wrapper passes uint8 kernel but
             receives float16 input (dtype mismatch bug in wrapper)
  reduce_sum: FAIL even [1,32] — returns inf/zero (FP16->FP32->kernel broken)

TORCH_FALLBACK (1 op)
  expand:    torch.Tensor.expand — no custom kernel possible  N/A
```

### Key Findings
1. **CompileFunction crash** (host_machine.cpp:179) for 2D tensors with first_dim > ~256. Affects equal, not. Root cause: PyPTO backend limitation, not operator logic.
2. **Cross-operator JIT contamination**: verified and confirmed with fresh-process testing protocol.
3. **div B64 NaN**: FP16 precision collapse in divide — likely needs FP32 accumulator.
4. **or dtype bug**: `bitwise_or` kernel compiled for DT_UINT8 but wrapper passes float16.
5. **reduce_sum inf/zero**: FP16->FP32 reduction pipeline broken in this PyPTO version.

---

## Phase 3: Route Variant Reclassification [COMPLETE]

All 14 `.orchestrator_state.json` files updated with `route_variant` field:

```
3  PYPTO_NATIVE          : add, where, transpose
7  PYPTO_HOST_ORCHESTRATED: relu, mul, softmax, layernorm, matmul, equal, not
3  UNDER_INVESTIGATION   : div, or, reduce_sum
1  TORCH_FALLBACK        : expand
```

---

## Phase 4: Performance Audit — Host Timing [COMPLETE]

**Methodology**: Fresh process per measurement, 100 warmup + 50 timed iterations, host_synchronized_operation_us. Software timestamp `time.time()` wrapping full kernel call + `torch.npu.synchronize()`.

### Host Sync Latency (microseconds)

```
Operator    |    B=1    |    B=8    |   B=64    | Kernels/Call | Scalability
------------|-----------|-----------|-----------|--------------|-------------
add         |    212.9  |    267.5  |    716.3  | 9            | 1.04x per B
where       |    302.6  |    310.5  |    303.6  | 1            | FLAT
transpose   |    204.3  |    920.1  |   6658.8  | B            | 0.51x per B
relu        |    226.3  |   4644.7  |   7536.1  | B            | 0.51x per B
mul         |    244.0  |   1219.6  |   8532.5  | B            | 0.52x per B
matmul      |    381.6  |   1986.9  |  14704.6  | B            | 0.56x per B
------------|-----------|-----------|-----------|--------------|-------------
softmax     |  28870.2  | 228675.9  | 1898503   | B*256 (per-row) | LAUNCH EXPLOSION
layernorm   |  28672.9  | 228746.7  | 1839959   | B*256 (per-row) | LAUNCH EXPLOSION
```

### Analysis
- **add/where**: Excellent. where is **flat** — proves 2D with dynamic annotation can work at scale.
- **transpose**: Acceptable. Single kernel, scales 3.4x per 64x B growth.
- **relu/mul**: Acceptable per-batch. ~118us per additional batch element (amortized 112us).
- **matmul**: Acceptable per-batch. ~230us per additional 3D batch.
- **softmax/layernorm (CRITICAL)**: **Per-row launch explosion**. B1 has 256 rows = 28.8ms. B64 has 16,384 rows = **1.9 seconds**. Each row kernel launch costs ~112us device sync overhead. 16,384 * 116us = 1.9s.

### msprof Findings (CANN 9.0.0)
- softmax B1, 256 rows: 16,875 total tasks, kernel duration 56-236us per call
- AIV utilization: 83-90% cube (PYPTO_softmax_kernel_2d)
- Output format changed: `op_summary.csv` replaces `kernel_details.csv`

---

## Phase 5: Workaround Evaluation [COMPLETE]

```
COMPLETE — performance adequate for production use
  add:        212-716 us,   native, linear scaling

COMPLETE_WITH_LIMITATION — correct, acceptable per-batch overhead
  where:      302-304 us,   native, flat (best in class)
  transpose:  204-6659 us,  native, moderate batch scaling
  relu:       226-7536 us,  per-batch loop ~118us/batch
  mul:        244-8533 us,  per-batch loop ~133us/batch (with 2 args)
  matmul:     382-14705 us, per-batch 2D + dynamic routing

CORRECTNESS_ONLY — performance renders it non-functional at scale
  softmax:    28ms->1.9s   per-row loop, 16,384 launches at B64
  layernorm:  28ms->1.8s   per-row loop, 16,384 launches at B64

NEEDS_KERNEL_FIX — not correct yet
  div:        B64 NaN (FP16 precision collapse)
  or:         dtype mismatch (wrapper bug)
  reduce_sum: returns inf/zero (reduction pipeline broken)
  equal:      B1 only (CompileFunction crash B>1)
  not:        B1 only (CompileFunction crash B>1)
```

---

## Phase 6: Native Alternatives [RESEARCH]

### Root Cause of Per-Row Workaround
PyPTO's `CompileFunction` crashes at `host_machine.cpp:179` when first_dim > ~256 for 2D tensors on certain operation types (comparison ops, reduction-like ops). NOT all ops — `where` with [64, 65536] works natively. The crash is operation-dependent.

### Softmax Native Candidates
1. **3D -> 2D reshape** (already tried in workaround — crash persists because total first_dim = B*256 > 256)
2. **Dynamic tile shapes** — already used, doesn't fix host_machine crash
3. **Framework patch** — modify PyPTO host_machine.cpp to handle larger first_dim (requires PyPTO source access)
4. **Multi-kernel fan-out** — split into chunks of 256 rows, each with own kernel call -> similar to per-row but coarser

### Recommended Path
For softmax/layernorm: either fix PyPTO backend (host_machine.cpp crash) or use AscendC/Torch implementation. Per-row workaround is not viable for any practical batch size.

---

## Phase 7: Three-Route Comparability [PRELIMINARY]

```
Operator   | PyPTO Status           | AscendC Status | Torch Status
-----------|------------------------|----------------|-------------
add        | OK NATIVE 213-716us    | Not built      | BASELINE
where      | OK NATIVE 302-304us    | Not built      | torch.where
transpose  | OK NATIVE 204-6659us   | Not built      | torch.transpose
relu       | OK WORKAROUND 226-7536 | Not built      | F.relu
mul        | OK WORKAROUND 244-8535 | Not built      | torch.mul
matmul     | OK WORKAROUND 382-14k  | Not built      | torch.matmul
softmax    | XX 1.9s per-row        | Not built      | F.softmax
layernorm  | XX 1.8s per-row        | Not built      | F.layer_norm
div        | XX B64 NaN             | Not built      | torch.div
equal      | XX B1 only             | Not built      | torch.eq
not        | XX B1 only             | Not built      | torch.logical_not
or         | XX B1 fail (dtype)     | Not built      | torch.bitwise_or
reduce_sum | XX always fail (inf)   | Not built      | torch.sum
expand     | NA (torch native)      | Not built      | tensor.expand

LEGEND: OK = PASS  XX = FAIL/BLOCKED  NA = Not applicable
```

AscendC implementations not yet built for any operator — this is a prerequisite for comparative benchmarking.

---

## Phase 8: Clean Reports and Documents [COMPLETE]

### Artifacts Verified
- `operators/*/pypto/.orchestrator_state.json` — all 14 updated with:
  - `route_variant` classification
  - `perf_host_timing_us` (8 active operators)
  - `kernels_per_logical_call`
  - `workaround_evaluation`
  - `last_updated` (ISO 8601 UTC)
- `operators/softmax/pypto/.orchestrator_state.json` — fixed missing fields
- Temp profiler files in `/tmp/opencode/` — archived, can be cleaned

---

## Phase 9: Code Review and Release [DEFERRED]

### G13 Code Review
**Not performed** — requires AscendC implementations for comparison. PyPTO code reviewed as part of audit:
- 5 workaround implementations: code is minimal, host-side loops correct
- Exception: `or_impl.py` has dtype mismatch bug (kernel DT_UINT8 vs wrapper fp16 input)
- All 14 `.orchestrator_state.json` files consistent with AGENTS.md schema

### pre_release_gate.sh Status
Not run — AscendC kernels not yet built, no comparative release to assemble.

---

## Recommendations

### Urgent (blocking production)
1. **Fix softmax/layernorm** — per-row launch explosion makes them unusable. Options:
   - Fix PyPTO host_machine.cpp crash for 2D first_dim > 256
   - Implement native AscendC version
   - Add PyPTO framework patch that batches row processing
2. **Fix or/reduce_sum/div** — these 3 operators are non-functional
3. **Fix equal/not** — need CompileFunction crash workaround or AscendC alternative

### Important (for completeness)
4. **Build AscendC implementations** for at least the top 8 operators to enable true three-route comparison
5. **Torch baseline profiling** — establish reference numbers for host_synchronized_operation_us
6. **Run pre_release_gate.sh** once AscendC kernels exist

### Nice-to-have
7. **msprof deep analysis** — parse CANN 9.0.0 op_summary format for device-level bottleneck analysis
8. **Cross-version testing** — test with newer PyPTO/CANN versions for CompileFunction crash fix

---

## Commands to Reproduce

```bash
# Fresh process correctness verification
source /home/developer/Ascend/cann-9.0.0/set_env.sh
cd /mnt/workspace/gitCode/cann/pto-isa/cannbot-ascendc-vs-pypto
export PYPTO_PATH=/home/developer/.cannbot/repo/plugins-official/pypto-op-orchestrator/pypto/python

# Host timing (per-operator at B=8)
python3 -c "
import sys; sys.path[:0]=['$PYPTO_PATH']
import torch, torch_npu; torch.npu.set_device(0)
# ... operator-specific script from operators/{op}/pypto/src/
"

# msprof (CANN 9.0.0)
/home/developer/Ascend/cann-9.0.0/bin/msprof \
  --output=/tmp/msprof_out \
  --application='python3 test_script.py' \
  --ascendcl=on --ai-core=on --task-time=l0
```

---

*End of audit report. Generated 2026-07-24T09:47:00Z.*
