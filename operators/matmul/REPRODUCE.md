# MatMul Operator — Reproduce

## Prerequisites
- Ascend 910B NPU
- CANN 9.0.0
- Python 3.8+ with PyTorch + torch_npu
- PyPTO 0.2.0

## Data Generation
```bash
cd operators/matmul
python3 data/generation_scripts/generate_inputs.py
python3 data/generation_scripts/generate_reference.py
```

## Torch Correctness
```bash
python3 torch/correctness.py
```

## Ascend C Build and Correctness
```bash
cd ascendc
cmake -S . -B build && cmake --build build -j
./build/matmul_ascendc 0 1 20 100 1000 10 operators/matmul/data build/output
```

## PyPTO Correctness
```bash
cd pypto
python3 test_matmul.py
```

## Benchmark
```bash
bash benchmark/run_all.sh
```
