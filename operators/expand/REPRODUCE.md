# Reproducing Expand Results

## Prerequisites
- Ascend 910B or compatible NPU
- CANN 9.0.0 (or compatible)
- torch, torch_npu
- PyPTO framework
- Cannbot Skills (for Ascend C)
- pypto-op-orchestrator (for PyPTO)

## Dependencies
See `/environment/environment_manifest.json` for exact versions.

## Steps

### 1. Environment Check
```bash
bash environment/preflight.sh
```

### 2. Generate Test Data
```bash
cd operators/expand
python3 data/generation_scripts/generate_inputs.py
python3 data/generation_scripts/generate_reference.py
```

### 3. Build Ascend C Binary
```bash
cd operators/expand/ascendc
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### 4. Check Correctness
```bash
# Torch baseline
python3 operators/expand/torch/correctness.py --batch 1,2,4,8,16,32,64

# Ascend C (will print per-batch results)
python3 operators/expand/ascendc/build/expand_ascendc

# PyPTO
python3 -c "
import sys; sys.path.insert(0, 'operators/expand/pyto/tests')
from test_expand import test_expand
test_expand()
"
```

### 5. Full Benchmark Run
```bash
bash operators/expand/benchmark/run_all.sh
```

### 6. Parse and Report
```bash
python3 scripts/generate_final_report.py
```

## Measurement Methodology

### Profiler-Based Device Kernel Measurement
All three implementations are measured using msprof with identical configuration:
- `--ascendcl=on --ai-core=on --task-time=l0`
- 200 warmup iterations (excluded from measurement)
- 100 profiled iterations

### Torch.expand
- Single msprof session: warmup(200) + loops(100) in same process
- Produces 1 KERNEL_AIVEC per call
- ACL context init in-session

### Ascend C Expand
- Single msprof session: warmup(200) + loops(100) via subprocess
- Produces 1 KERNEL_AIVEC per call
- ACL context + kernel load in-session

### PyPTO Expand
- Two-process method: warmup(200) no-profiler, then msprof session for loops(100)
- Produces 3 kernels per call: 1 KERNEL_MIX_AIC (compute) + 2 KERNEL_AICPU (auxiliary)
- JIT compiled in warmup process before profiler starts

## Metrics
- `all_device_kernels_us`: Sum of all KERNEL_* durations per logical call
- `primary_compute_kernel_us`: Duration of the main compute kernel
- `kernels_per_call`: Number of device kernel events per logical call
- `host_synchronized_operation_us`: Operation-level latency from host perspective

## Known Limitations
1. PyPTO's 3 kernel events may overlap in hardware (pipelining). The all_device_kernels sum is an upper bound; actual wall-clock time may be lower.
2. PyPTO's torch.npu.Event measurement captures the actual end-to-end device wall-clock time accounting for overlap.
3. Comparison is valid only for the specific shape [B,256,384] FP16 on Ascend 910B.
