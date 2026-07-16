# Reproducing Mul Experiment

## Prerequisites
- Ascend 910B NPU (device 0)
- CANN 9.0.0
- Python 3.11
- torch + torch_npu (Ascend compatible)
- PyPTO

## Steps

### 1. Environment check
```bash
bash ../../environment/preflight.sh
```

### 2. Generate test data
```bash
python3 data/generation_scripts/generate_inputs.py
python3 data/generation_scripts/generate_reference.py
```

### 3. Build Ascend C binary
```bash
mkdir -p ascendc/build
cmake -S ascendc -B ascendc/build
cmake --build ascendc/build -j
```

### 4. Verify correctness
```bash
# Torch
python3 torch/correctness.py --batch "1,2,4,8,16,32,64"

# Ascend C (all batches)
for b in 1 2 4 8 16 32 64; do
  ./ascendc/build/mul_ascendc 0 $b 20 8192 1 1 1 data ascendc/build/output
  python3 data/generation_scripts/correctness.py ascendc/build/output/output_b${b}.bin data/reference_b${b}_fp16.bin $b
done

# PyPTO
python3 pypto/tests/test_mul.py
```

### 5. Run profiler benchmarks
```bash
bash benchmark/run_all.sh
```

Or individually:

**Torch**:
```bash
msprof --output=reports/raw/torch_b1 --ascendcl=on --ai-core=on --task-time=l0 \
  python3 torch/benchmark.py --batch 1 --warmup 200 --loops 100 --repeat 5
```

**Ascend C**:
```bash
msprof --output=reports/raw/ascendc_b1 --ascendcl=on --ai-core=on --task-time=l0 \
  ./ascendc/build/mul_ascendc 0 1 20 8192 200 100 5 data ascendc/build/output
```

**PyPTO** (two-process):
```bash
# Warmup (no profiler)
python3 -c "
import sys; sys.path.insert(0, 'pypto/src'); sys.path.insert(0, 'pypto/golden')
from mul_impl import mul_wrapper
import torch; import torch_npu
x1=torch.randn(1,3,4,256,32,dtype=torch.float16).npu()
x2=torch.randn(1,3,4,256,32,dtype=torch.float16).npu()
for _ in range(200): mul_wrapper(x1,x2)
torch.npu.synchronize()
"
# Measure
msprof --output=reports/raw/pypto_b1 --ascendcl=on --ai-core=on --task-time=l0 \
  python3 -c "
import sys; sys.path.insert(0, 'pypto/src'); sys.path.insert(0, 'pypto/golden')
from mul_impl import mul_wrapper
import torch; import torch_npu
x1=torch.randn(1,3,4,256,32,dtype=torch.float16).npu()
x2=torch.randn(1,3,4,256,32,dtype=torch.float16).npu()
for _ in range(100): mul_wrapper(x1,x2)
torch.npu.synchronize()
"
```

### 6. Parse and report
```bash
python3 benchmark/parse_profiler.py reports/raw/torch_b1 reports/parsed/torch_b1.json
python3 benchmark/parse_profiler.py reports/raw/ascendc_b1 reports/parsed/ascendc_b1.json
python3 benchmark/parse_profiler.py reports/raw/pypto_b1 reports/parsed/pypto_b1.json
```

## Artifacts

All output files (Ascend C output binaries, parsed profiler data, correctness results) are stored within `operators/mul/`. SHA256SUMS at the operator root for integrity verification.
