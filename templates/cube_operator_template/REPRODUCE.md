# Reproducing Results

## Prerequisites
- Ascend 910B NPU
- CANN 9.0.0+
- Python 3.8+ with PyTorch + torch_npu
- PyPTO framework (version 0.2.0+)

## Steps
1. Generate data: `python3 data/generation_scripts/generate_inputs.py && python3 data/generation_scripts/generate_reference.py`
2. Torch correctness: `python3 torch/correctness.py`
3. Build Ascend C: `cd ascendc && cmake -S . -B build && cmake --build build -j`
4. Ascend C correctness: `python3 data/generation_scripts/correctness.py ascendc`
5. PyPTO correctness: `python3 pypto/test_matmul.py`
6. Benchmark: `bash benchmark/run_all.sh`
