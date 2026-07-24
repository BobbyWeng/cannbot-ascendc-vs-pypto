# Perf Data Provenance

## Host Timing (perf_2026-07-24/host_timing_results.json)
- Source script: perf_2026-07-24/perf_measurement_script.py
- Method: `perf_simple2.py` run at 2026-07-24T09:43Z
- 8 active operators: add, where, transpose, relu, mul, softmax, layernorm, matmul
- Batches: B=[1, 8, 64]
- Warmup: 100 iterations, Timed: 50 iterations
- Metric: `host_synchronized_operation_us` (time.time() + torch.npu.synchronize())

## msprof (perf_2026-07-24/msprof/softmax_b1/)
- Tool: msprof (CANN 9.0.0, /home/developer/Ascend/cann-9.0.0/bin/msprof)
- Flags: --ascendcl=on --ai-core=on --task-time=l0
- Operator: softmax B=1, shape [256,384] fp16
- Output format: CANN 9.0.0 mindstudio_profiler_output (op_summary.csv replaces kernel_details.csv)

## Raw msprof Data (NOT committed due to size, 7.7G)
- Path on disk: reports/audit/perf_2026-07-23/raw/where/
- Batches collected: B4, B8, B32, B64
- Collected 2026-07-23 for where operator
- NOT committed to git due to size constraints (7.7GB of raw .db/.data files)
- Can be regenerated using scripts in operators/where/benchmark/

## Environment
- torch: 2.8.0+cpu, torch_npu: 2.8.0.post2
- CANN: 9.0.0 (source /home/developer/Ascend/cann-9.0.0/set_env.sh)
- PyPTO: d1c290f3691effe350243f49acb0b262d0ca2e39
- NPU: Ascend910 (npu-smi verified)
- Device: 0, Platform: linux x86_64
