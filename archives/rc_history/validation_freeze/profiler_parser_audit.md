# Profiler Parser Audit — Validation Freeze

## Parser Methodology

The project uses `common/profiler/parse_profiler.py` which:
1. Finds the `PROF_*` directory in raw msprof output
2. Locates `task_time*.csv` and `msprof_*.json` (Chrome trace)
3. Filters for `ph: "X"` (complete) events with `KERNEL_*` Task Type
4. Groups by kernel type, identifies primary (longest) kernel
5. Divides total duration by loops (100) for per-call metrics

## Metric Hierarchy Verification

The parser correctly implements:
1. **Primary compute kernel** (`primary_compute_kernel_us`): Duration of single longest kernel event
2. **All device kernels** (`all_device_kernels_us_per_call`): Sum of all kernel durations / loops
3. **Kernel type breakdown** (`kernel_type_breakdown`): Per-type statistics

The parser does NOT mix levels. All msprof reports use device-kernel timing.

## Coverage

| Operator | Parsed Files | Primary Kernel Identified? | Kernel Type Correct? |
|----------|-------------|---------------------------|---------------------|
| relu | 21 (7x3) | YES | AIVEC AIC AICPU |
| mul | 21 (7x3) | YES | AIVEC |
| add | 21 (7x3) | YES | AIVEC + AICPU |
| div | 12 (6x2) | YES | AIVEC |
| matmul | 12 (6x2) | YES | AICORE |
| expand | 21 (7x3) | YES | AIVEC + AICPU |
| transpose | 14 (7x2) | YES | AIVEC |
| reduce_sum | 12 (6x2) | YES | AIVEC |
| equal | 0 | NO | Event-only, no msprof |
| not | 0 | NO | Event-only, no msprof |
| or | 0 | NO | Event-only, no msprof |
| where | 0 | NO | Event-only, no msprof |

## Critical: Event-Based Operators Lack Parser Data

Operators equal, not, or, where have **zero parsed msprof files**. Their raw data consists of `result.json` files from `torch.npu.Event` timing, which measures host-synchronized latency, not device kernel duration. The parser does not process these files.

For these operators:
- `kernel_names` = `['unknown']` (no msprof trace)
- No `primary_compute_kernel_us` available
- Reports correctly mark them as Event-based

## Parser Quality Issues

1. **Kernel name deduplication**: The parser lists kernel names in `kernel_names` array. For multi-kernel calls (e.g., torch.add with 3 chained calls), all 3 kernel names are captured. This is correct.

2. **Kernel count per call**: The parser correctly computes `kernels_per_logical_call = total_events / loops`. For torch.add, this counts all 21 device kernel events (7 per call x 3 calls).

3. **Primary kernel identification**: Uses `max()` on event durations. Correct for single-kernel calls; for multi-kernel calls, returns the longest single event, which is the correct primary compute kernel.

4. **Two-process JIT handling**: PyPTO uses a separate warmup process (no profiler) then msprof session. The parser correctly excludes JIT compilation overhead.

## Recommendations

1. **Run msprof for equal, not, or, where** to produce comparable device-kernel timing
2. **Verify parser against raw Chrome trace events** for one operator per group (AIVEC, AICORE, AICPU)
