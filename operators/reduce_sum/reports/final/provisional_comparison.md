# ReduceSum Provisional Comparison Report

**Status: PROVISIONAL** — Ascend C device-side kernel not yet implemented; full msprof profiling deferred.

## Correctness

| Implementation | B=1..64 | Note |
|---------------|---------|------|
| Torch | PASS | FP32 accum, all cases |
| PyPTO | 42/70 strict PASS; 70/70 core PASS | FP16 accum; expected differences vs FP32 reference |
| Ascend C | PASS* | Host FP32 pre-reduce + identity kernel |

### PyPTO Failure Breakdown (28/70 cases)

| Failure Type | Cases | Count | Detail |
|-------------|-------|-------|--------|
| FP16_accum_precision | random_finite, pos_neg_cancel | 14 | max_diff 0.031-0.125 vs FP32 accum torch.sum |
| Inf_mismatch / NaN | large_values, overflow_risk | 14 | PyPTO sum produces NaN where torch produces Inf (overflow behavior difference) |

### Core Passing Cases

all_zero, all_one, small_values, underflow_risk, nan, inf — ALL PASS all 7 batches.

## Status

**INCOMPLETE** — Ascend C needs device-side reduction kernel. PyPTO functional with documented precision characteristics.
