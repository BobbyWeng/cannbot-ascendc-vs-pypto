#!/bin/bash
# Benchmark runner for ReduceSum
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(dirname "$SCRIPT_DIR")"
PROJ_DIR="$(dirname "$(dirname "$OP_DIR")")"

if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    source "$ASCEND_HOME_PATH/set_env.sh"
fi

DATA_DIR="$OP_DIR/data"
if [ ! -f "$DATA_DIR/x_b1_random_finite.bin" ]; then
    echo "Generating data..."
    python3 "$DATA_DIR/generation_scripts/generate_inputs.py"
    python3 "$DATA_DIR/generation_scripts/generate_reference.py"
fi

if [ ! -f "$OP_DIR/ascendc/build/reduce_sum_ascendc" ]; then
    echo "Building Ascend C..."
    mkdir -p "$OP_DIR/ascendc/build"
    cmake -S "$OP_DIR/ascendc" -B "$OP_DIR/ascendc/build"
    cmake --build "$OP_DIR/ascendc/build" -j
fi

echo "=== Torch Baseline ==="
python3 "$OP_DIR/torch/correctness.py"
python3 "$OP_DIR/torch/benchmark.py"

echo "=== Ascend C Correctness ==="
mkdir -p "$OP_DIR/ascendc/build/output"
for b in 1 2 4 8 16 32 64; do
    for case in random_finite all_zero all_one pos_neg_cancel small_values large_values overflow_risk underflow_risk nan inf; do
        "$OP_DIR/ascendc/build/reduce_sum_ascendc" 0 $b 20 8192 1 1 1 "$DATA_DIR" "$OP_DIR/ascendc/build/output" "$case"
    done
done
python3 "$DATA_DIR/generation_scripts/correctness.py"

echo "=== PyPTO Correctness ==="
python3 "$OP_DIR/pypto/correctness.py"

echo "=== Ascend C Profiler ==="
for b in 1 2 4 8 16 32 64; do
    msprof --output="$OP_DIR/reports/raw/ascendc_b${b}" \
           --ascendcl=on --ai-core=on --task-time=l0 \
           "$OP_DIR/ascendc/build/reduce_sum_ascendc" 0 $b 20 8192 200 100 5 "$DATA_DIR" "$OP_DIR/ascendc/build/output" "random_finite"
done

echo "=== Torch Profiler ==="
for b in 1 2 4 8 16 32 64; do
    msprof --output="$OP_DIR/reports/raw/torch_b${b}" \
           --ascendcl=on --ai-core=on --task-time=l0 \
           python3 -c "
import torch, torch_npu
x = torch.randn($b, 256, 384, dtype=torch.float16).npu()
for _ in range(200): torch.sum(x, dim=-1)
torch.npu.synchronize()
for _ in range(100): torch.sum(x, dim=-1)
torch.npu.synchronize()
"
done

echo "=== All benchmarks complete ==="
