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
BLOCK_DIM=20
TILE_LEN=8192
DATA_DIR="$OPER_DIR/data"
ASCENDC_BIN="$OPER_DIR/ascendc/build/layernorm_ascendc"
RAW_DIR="$OPER_DIR/reports/raw"
PARSED_DIR="$OPER_DIR/reports/parsed"
mkdir -p "$RAW_DIR" "$PARSED_DIR"
source ~/Ascend/ascend-toolkit/set_env.sh

echo "=== Ascend C msprof all batches ==="
for b in "${BATCHES[@]}"; do
    echo "[BENCH] batch=$b"
    label="ascendc_b${b}"
    out_dir="$RAW_DIR/${label}"
    parsed="$PARSED_DIR/${label}.json"
    mkdir -p "$out_dir"
    msprof --output="$out_dir" --ascendcl=on --ai-core=on --task-time=l0 \
        "$ASCENDC_BIN" "$DEVICE_ID" "$b" "$BLOCK_DIM" "$TILE_LEN" \
            "$WARMUP" "$LOOPS" "$REPEAT" "$DATA_DIR" "$OPER_DIR/ascendc/build/output" > /dev/null 2>&1
    python3 "$BENCH_DIR/parse_profiler.py" "$out_dir" "$parsed"
done

echo "=== PyPTO msprof all batches ==="
PYPTO_SCRIPT=$(mktemp)
cat > "$PYPTO_SCRIPT" << 'PYEOF'
import os, sys, json, torch, numpy as np
import torch_npu
sys.path.insert(0, sys.argv[1])
from layernorm_impl import layernorm_wrapper
B = int(sys.argv[2])
DATA_DIR = sys.argv[3]
x = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f'input_b{B}_fp16.bin'), dtype=np.float16).reshape(B, 256, 32))
w_t = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, 'weight_fp16.bin'), dtype=np.float16).reshape(-1))
b_t = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, 'bias_fp16.bin'), dtype=np.float16).reshape(-1))
torch.npu.set_device(0)
x_npu = x.npu(); w_npu = w_t.npu(); b_npu = b_t.npu()
for _ in range(200):
    _ = layernorm_wrapper(x_npu, w_npu, b_npu)
torch.npu.synchronize(0)
for _ in range(100):
    _ = layernorm_wrapper(x_npu, w_npu, b_npu)
torch.npu.synchronize(0)
PYEOF
chmod +x "$PYPTO_SCRIPT"
for b in "${BATCHES[@]}"; do
    echo "[BENCH] PyPTO batch=$b"
    label="pypto_b${b}"
    out_dir="$RAW_DIR/${label}"
    parsed="$PARSED_DIR/${label}.json"
    mkdir -p "$out_dir"
    msprof --output="$out_dir" --ascendcl=on --ai-core=on --task-time=l0 \
        python3 "$PYPTO_SCRIPT" "$OPER_DIR/pypto/src" "$b" "$DATA_DIR" > /dev/null 2>&1
    python3 "$BENCH_DIR/parse_profiler.py" "$out_dir" "$parsed"
done
rm -f "$PYPTO_SCRIPT"
echo "=== All done ==="
