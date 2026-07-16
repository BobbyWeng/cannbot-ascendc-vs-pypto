# Reproducing Where Experiment

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
  ./ascendc/build/where_ascendc 0 $b 20 8192 1 1 1 data ascendc/build/output
  python3 data/generation_scripts/correctness.py ascendc/build/output/output_b${b}.bin data/reference_b${b}_fp16.bin $b
done

# PyPTO
python3 pypto/correctness.py --batch "1,2,4,8,16,32,64"
```

### 5. Run profiler benchmarks
```bash
bash benchmark/run_all.sh
```

### 6. View Results
```bash
cat reports/final/final_comparison.md
```

## Measurement Methodology
All three implementations measured using msprof with identical configuration:
- `--ascendcl=on --ai-core=on --task-time=l0`
- 200 warmup iterations (excluded)
- 100 profiled iterations
- 5 repeats

## Artifacts
All output files are stored within `operators/where/`. SHA256SUMS at the operator root for integrity verification.
