# Profiler & Benchmark Unification Issues

## Methodology Violations

### 1. Equal: No msprof profiling performed
- **Claim**: experiment_config.yaml says "profiler: msprof" with single msprof session
- **Reality**: Torch uses torch.npu.Event, Ascend C uses aclrtEventElapsedTime. No msprof raw data exists.
- **Reports**: `local_gate.json` confirms profiler: NOT_RUN for both torch and ascendc
- **Impact**: Latency numbers in final reports are host-synchronized operation times, NOT primary_compute_kernel_us as required by AGENTS.md

### 2. Not: No msprof profiling performed
- **Claim**: REPRODUCE.md says "All three implementations measured using msprof"
- **Reality**: Torch benchmark uses torch.npu.Event, not msprof. Ascend C uses aclrtEvent timing.
- **No reports/parsed/ directory** exists
- **Raw data** is JSON result files, not msprof trace output
- `local_gate.json` says profiler: NOT_RUN

### 3. Or: No msprof profiling performed
- **Same issues as Not**
- benchmark/run_all.sh exists but SKIPS both Ascend C and PyPTO sections
- `local_gate.json` says profiler: NOT_RUN

### 4. Where: No msprof profiling performed
- **Same pattern**: Uses torch.npu.Event and aclrtEvent, not msprof
- `local_gate.json` says profiler: NOT_RUN
- Ascend C profiler numbers show suspiciously perfect deterministic timing (identical repeat values), consistent with non-vectorized scalar loop

## Standard Violations

### 5. Warmup inconsistency
| Operator | Torch | Ascend C | Standard |
|----------|-------|----------|----------|
| relu | 200 | 100 in host | 200 |
| mul | 200 | 100 in host | 200 |
| add | 200 | 100 in host | 200 |
| div | 200 | 100 in host | 200 |
| equal | N/A (no msprof) | N/A | 200 |
| not | N/A (no msprof) | N/A | 200 |
| or | N/A (no msprof) | N/A | 200 |
| where | N/A (no msprof) | N/A | 200 |

### 6. Profiled loops inconsistency
| Operator | Torch | Ascend C | Standard |
|----------|-------|----------|----------|
| relu | 100 | 1000 in host | >= 100 |
| mul | 100 | 1000 in host | >= 100 |
| add | 100 | 1000 in host | >= 100 |
| div | 100 | 1000 in host | >= 100 |

### 7. Host activity latency reported as device kernel time
- **Not**: Ascend C Not B=1 is 6.4 us — likely aclrtEvent device-kernel-level timing (not msprof)
- **Not**: PyPTO Not flat ~130 us across all batch sizes — likely host dispatch overhead, not device kernel

### 8. PyPTO JIT handling unknown
- **Not**: Two-process method claimed but unverifiable (no msprof raw data)
- **Or**: Two-process method claimed but unverifiable
- **Where**: N/A (BLOCKED_BACKEND)

### 9. missing all_device_kernels_us_per_call in Not/Or/Where/Div reports
- Not and Or final reports only have host-timed latency, no msprof kernel breakdown
- Div has msprof for B=32 only

### 10. Kernel count not verified across implementations
- Not final report: kernel identity marked "unknown" for PyPTO
- Or final report: kernel identity marked "unknown" for PyPTO

## Report vs Profiler Discrepancies

### 11. Add final_comparison.json PyPTO primary kernel mislabeled
- Parsed (pypto_b1): KERNEL_MIX_AIC mean = 51.128 us (correct primary)
- Final report: primary_compute_kernel_us = 3161.403 (wrong — this is KERNEL_AICPU init)
- The same issue exists for release_summary.json

### 12. Div latency discrepancy
- Ascend C B=32: aclrtEvent = 328.6 us, msprof = 306.5 us (7.2% error)
- Torch B=32: torch.Event = 112.80 us, msprof = 126.20 us (11.9% error)
- Both documented in prior audit but report needs correction

### 13. Release summary primary kernel data is wrong for PyPTO
- release_summary.json shows PyPTO primary_compute_kernel_us as KERNEL_AICPU (e.g. 3027.82 us for relu B=1)
- The actual KERNEL_MIX_AIC primary is ~51-98 us
- This makes PyPTO look ~30x slower than it actually is for compute
