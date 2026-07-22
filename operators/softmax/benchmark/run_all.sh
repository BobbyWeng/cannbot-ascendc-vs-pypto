#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="$(dirname "$0")"
OPER_DIR="$(realpath "$BENCH_DIR/..")"
PROJECT_ROOT="$(realpath "$OPER_DIR/../..")"

BATCHES=(1 2 4 8 16 32 64)
WARMUP=200
LOOPS=100
REPEAT=5
DEVICE_ID=0
DTYPE=fp16

DATA_DIR="$OPER_DIR/data"
ASCENDC_DIR="$OPER_DIR/ascendc"
ASCENDC_BUILD_DIR="$ASCENDC_DIR/build"
ASCENDC_BIN="$ASCENDC_BUILD_DIR/softmax_ascendc"
TORCH_DIR="$OPER_DIR/torch"
PYPTO_DIR="$OPER_DIR/pypto"
REPORTS_DIR="$OPER_DIR/reports"
RAW_DIR="$REPORTS_DIR/raw"
PARSED_DIR="$REPORTS_DIR/parsed"

mkdir -p "$RAW_DIR" "$PARSED_DIR"

echo "============================================"
echo " Softmax Benchmark Runner — Measurement Contract Uniform"
echo "============================================"

# Source CANN
if [ -f ~/Ascend/ascend-toolkit/set_env.sh ]; then
    source ~/Ascend/ascend-toolkit/set_env.sh
fi

bench_torch() {
    local batch=$1
    local label="torch_b${batch}"
    local out_dir="$RAW_DIR/${label}"
    local parsed="$PARSED_DIR/${label}.json"
    echo "[BENCH] Torch batch=$batch"
    msprof --output="$out_dir" --ascendcl=on --ai-core=on --task-time=l0 \
        python3 "$TORCH_DIR/benchmark.py" \
            --batch "$batch" --warmup "$WARMUP" --loops "$LOOPS" \
            --repeat "$REPEAT" --device "$DEVICE_ID"
    python3 "$BENCH_DIR/parse_profiler.py" "$out_dir" "$parsed"
}

bench_pypto() {
    local batch=$1
    local label="pypto_b${batch}"
    local out_dir="$RAW_DIR/${label}"
    local parsed="$PARSED_DIR/${label}.json"
    echo "[BENCH] PyPTO batch=$batch (two-process)"

    PYPTO_BENCH_SCRIPT=$(mktemp)
    cat > "$PYPTO_BENCH_SCRIPT" << 'PYEOF'
import os, sys, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
from softmax_impl import softmax_wrapper

batch = int(sys.argv[1])
warmup = int(sys.argv[2])
loops = int(sys.argv[3])
repeat = int(sys.argv[4])
device_id = int(sys.argv[5])
data_dir = sys.argv[6]

shape = [batch, 256, 32]
in_path = os.path.join(data_dir, f"input_b{batch}_fp16.bin")
x = torch.from_numpy(np.fromfile(in_path, dtype=np.float16).reshape(shape))
import torch_npu
torch.npu.set_device(device_id)
x_npu = x.npu(device_id)

for _ in range(warmup):
    _ = softmax_wrapper(x_npu)
torch.npu.synchronize(device_id)

for _ in range(loops + repeat):
    _ = softmax_wrapper(x_npu)
torch.npu.synchronize(device_id)
PYEOF

    chmod +x "$PYPTO_BENCH_SCRIPT"

    # Phase 1: warmup without profiler
    echo "  [phase 1] warmup..."
    python3 "$PYPTO_BENCH_SCRIPT" "$batch" "$WARMUP" 0 0 "$DEVICE_ID" "$DATA_DIR"

    # Phase 2: msprof for loops only
    echo "  [phase 2] msprof measurement..."
    msprof --output="$out_dir" --ascendcl=on --ai-core=on --task-time=l0 \
        python3 "$PYPTO_BENCH_SCRIPT" "$batch" 0 "$LOOPS" "$REPEAT" "$DEVICE_ID" "$DATA_DIR"

    rm -f "$PYPTO_BENCH_SCRIPT"
    python3 "$BENCH_DIR/parse_profiler.py" "$out_dir" "$parsed"
}

# ---- Torch all batches ----
for b in "${BATCHES[@]}"; do
    bench_torch "$b"
done

# ---- PyPTO B=1 ----
bench_pypto 1

echo ""
echo "=== All benchmarks complete ==="
echo "Torch batches: ${BATCHES[*]}"
echo "PyPTO: B=1"
echo "Raw: $RAW_DIR"
echo "Parsed: $PARSED_DIR"
