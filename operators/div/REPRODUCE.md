# Reproducing Div Broadcast Experiment

## Prerequisites
- Ascend 910B NPU
- CANN Toolkit
- Python 3.8+ with PyTorch + torch_npu
- PyPTO framework
- Cannbot Skills

## Step-by-step

### 1. Environment
```bash
source /etc/profile.d/ascend_env.sh
source ~/Ascend/ascend-toolkit/set_env.sh
```

### 2. Generate Data
```bash
cd operators/div
python3 data/generation_scripts/generate_inputs.py
python3 data/generation_scripts/generate_reference.py
```

### 3. Build Ascend C
```bash
cd operators/div/ascendc
mkdir -p build && cd build
cmake .. && cmake --build . -j
```

### 4. Run Full Benchmark
```bash
cd operators/div
bash benchmark/run_all.sh
```

### 5. View Results
```bash
cat reports/final/final_comparison.md
```
