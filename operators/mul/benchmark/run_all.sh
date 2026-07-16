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
DTYPE=fp16

DATA_DIR="$OPER_DIR/data"
ASCENDC_DIR="$OPER_DIR/ascendc"
ASCENDC_BUILD_DIR="$ASCENDC_DIR/build"
ASCENDC_BIN="$ASCENDC_BUILD_DIR/mul_ascendc"
TORCH_DIR="$OPER_DIR/torch"
PYPTO_DIR="$OPER_DIR/pypto"
REPORTS_DIR="$OPER_DIR/reports"
RAW_DIR="$REPORTS_DIR/raw"
PARSED_DIR="$REPORTS_DIR/parsed"
CORRECTNESS_DIR="$REPORTS_DIR/correctness"
FINAL_DIR="$REPORTS_DIR/final"

mkdir -p "$RAW_DIR" "$PARSED_DIR" "$CORRECTNESS_DIR" "$FINAL_DIR"

echo "============================================"
echo " Mul Benchmark Runner"
echo "============================================"

# ------------------------------------------------------------------
# 1. Source CANN environment
# ------------------------------------------------------------------
if [ -f /etc/profile.d/ascend_env.sh ]; then
    source /etc/profile.d/ascend_env.sh
fi
if [ -f ~/Ascend/ascend-toolkit/set_env.sh ]; then
    source ~/Ascend/ascend-toolkit/set_env.sh
fi
if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    echo "[OK] CANN found at $ASCEND_HOME_PATH"
else
    echo "[WARN] ASCEND_HOME_PATH not set"
fi

# ------------------------------------------------------------------
# 2. Preflight checks
# ------------------------------------------------------------------
echo ""
echo "--- Preflight checks ---"

check_data() {
    local missing=0
    for b in "${BATCHES[@]}"; do
        if [ ! -f "$DATA_DIR/x1_b${b}_${DTYPE}.bin" ]; then echo "[MISS] x1_b${b}_${DTYPE}.bin"; missing=1; fi
        if [ ! -f "$DATA_DIR/x2_b${b}_${DTYPE}.bin" ]; then echo "[MISS] x2_b${b}_${DTYPE}.bin"; missing=1; fi
        if [ ! -f "$DATA_DIR/reference_b${b}_${DTYPE}.bin" ]; then echo "[MISS] reference_b${b}_${DTYPE}.bin"; missing=1; fi
    done
    return $missing
}

if check_data; then
    echo "[OK] All data files present"
else
    echo "[GEN] Generating data files..."
    python3 "$DATA_DIR/generation_scripts/generate_inputs.py"
    python3 "$DATA_DIR/generation_scripts/generate_reference.py"
fi

if [ -x "$ASCENDC_BIN" ]; then
    echo "[OK] Ascend C binary found"
else
    echo "[BUILD] Building Ascend C binary..."
    mkdir -p "$ASCENDC_BUILD_DIR"
    cmake -S "$ASCENDC_DIR" -B "$ASCENDC_BUILD_DIR"
    cmake --build "$ASCENDC_BUILD_DIR" -j
    if [ ! -x "$ASCENDC_BIN" ]; then
        echo "[FAIL] Ascend C build failed"
        exit 1
    fi
    echo "[OK] Ascend C build complete"
fi

echo "[CHECK] PyPTO imports..."
python3 -c "
import sys
sys.path.insert(0, '$PYPTO_DIR/src')
sys.path.insert(0, '$PYPTO_DIR/golden')
from mul_golden import mul_golden
from mul_impl import mul_wrapper
print('[OK] PyPTO imports successful')
" 2>&1 || echo "[WARN] PyPTO import check failed"

# ------------------------------------------------------------------
# 3. Correctness verification
# ------------------------------------------------------------------
echo ""
echo "--- Correctness verification ---"

BATCH_CSV=$(IFS=,; echo "${BATCHES[*]}")

echo "[CORRECT] Torch baseline..."
python3 "$TORCH_DIR/correctness.py" --batch "$BATCH_CSV" --device "$DEVICE_ID" 2>&1
cp "$TORCH_DIR/correctness_results.json" "$CORRECTNESS_DIR/torch_correctness.json"

echo "[CORRECT] Ascend C..."
ASCENDC_OUT_DIR="$ASCENDC_BUILD_DIR/output"
mkdir -p "$ASCENDC_OUT_DIR"
ASC_CORRECT_PASS=true
for b in "${BATCHES[@]}"; do
    "$ASCENDC_BIN" "$DEVICE_ID" "$b" "$BLOCK_DIM" "$TILE_LEN" 1 1 1 "$DATA_DIR" "$ASCENDC_OUT_DIR" > /dev/null 2>&1
    python3 "$DATA_DIR/generation_scripts/correctness.py" \
        "$ASCENDC_OUT_DIR/output_b${b}.bin" \
        "$DATA_DIR/reference_b${b}_${DTYPE}.bin" \
        "$b" || ASC_CORRECT_PASS=false
done

echo "[CORRECT] PyPTO..."
python3 "$PYPTO_DIR/tests/test_mul.py" 2>&1 || true

# ------------------------------------------------------------------
# 4. Profiler benchmarks
# ------------------------------------------------------------------
echo ""
echo "--- Profiler benchmarks ---"

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

bench_ascendc() {
    local batch=$1
    local label="ascendc_b${batch}"
    local out_dir="$RAW_DIR/${label}"
    local parsed="$PARSED_DIR/${label}.json"
    echo "[BENCH] Ascend C batch=$batch"
    msprof --output="$out_dir" --ascendcl=on --ai-core=on --task-time=l0 \
        "$ASCENDC_BIN" "$DEVICE_ID" "$batch" "$BLOCK_DIM" "$TILE_LEN" \
            "$WARMUP" "$LOOPS" "$REPEAT" "$DATA_DIR" "$ASCENDC_OUT_DIR"
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
from mul_impl import mul_wrapper

batch = int(sys.argv[1])
warmup = int(sys.argv[2])
loops = int(sys.argv[3])
repeat = int(sys.argv[4])
device_id = int(sys.argv[5])
data_dir = sys.argv[6]

shape_tail = [3, 4, 256, 32]
shape = [batch] + shape_tail
x1_path = os.path.join(data_dir, f"x1_b{batch}_fp16.bin")
x2_path = os.path.join(data_dir, f"x2_b{batch}_fp16.bin")
x1 = torch.from_numpy(np.fromfile(x1_path, dtype=np.float16).reshape(shape))
x2 = torch.from_numpy(np.fromfile(x2_path, dtype=np.float16).reshape(shape))
import torch_npu
torch.npu.set_device(device_id)
x1_npu = x1.npu(device_id)
x2_npu = x2.npu(device_id)

for _ in range(warmup):
    _ = mul_wrapper(x1_npu, x2_npu)
torch.npu.synchronize(device_id)

for _ in range(loops + repeat):
    _ = mul_wrapper(x1_npu, x2_npu)
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

for b in "${BATCHES[@]}"; do
    bench_torch "$b"
    bench_ascendc "$b"
    bench_pypto "$b"
done

# ------------------------------------------------------------------
# 5. Aggregate parsed results
# ------------------------------------------------------------------
echo ""
echo "--- Aggregating results ---"
python3 -c "
import os, json
from collections import defaultdict

parsed_dir = '$PARSED_DIR'
batches = [1, 2, 4, 8, 16, 32, 64]
impls = ['torch', 'ascendc', 'pypto']

aggregate = {}
for impl in impls:
    aggregate[impl] = {}
    for b in batches:
        fpath = os.path.join(parsed_dir, f'{impl}_b{b}.json')
        if os.path.exists(fpath):
            with open(fpath) as f:
                aggregate[impl][str(b)] = json.load(f)

out = os.path.join('$PARSED_DIR', 'aggregate.json')
with open(out, 'w') as f:
    json.dump(aggregate, f, indent=2)
print(f'aggregate -> {out}')
"

# ------------------------------------------------------------------
# 6. Generate final report
# ------------------------------------------------------------------
echo ""
echo "--- Final report generation ---"
python3 -c "
import os, json, sys
sys.path.insert(0, '$PROJECT_ROOT')
from common.reporting import generate_markdown_report, generate_json_report

parsed_dir = '$PARSED_DIR'
final_dir = '$FINAL_DIR'
batches = [1, 2, 4, 8, 16, 32, 64]
impls = ['torch', 'ascendc', 'pypto']

aggregate = {}
fpath = os.path.join(parsed_dir, 'aggregate.json')
if os.path.exists(fpath):
    with open(fpath) as f:
        aggregate = json.load(f)

report = {
    'experiment': {
        'operator': 'mul',
        'formula': 'Y = X1 * X2',
        'shape': '[B, 3, 4, 256, 32]',
        'dtype': 'float16',
        'batches': batches,
        'warmup': $WARMUP,
        'loops': $LOOPS,
        'repeat': $REPEAT,
        'benchmark': 'msprof',
    },
    'batches': {},
}

for b in batches:
    key = str(b)
    batch_report = {}
    for impl in impls:
        if impl in aggregate and key in aggregate[impl]:
            batch_report[impl] = aggregate[impl][key]
    report['batches'][key] = batch_report

json_out = os.path.join(final_dir, 'benchmark_report.json')
generate_json_report(report, json_out)

md_out = os.path.join(final_dir, 'benchmark_report.md')
generate_markdown_report(report, md_out)

print(f'json -> {json_out}')
print(f'md  -> {md_out}')
"

# ------------------------------------------------------------------
# 7. Print summary
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo " Summary"
echo "============================================"
echo " Operator  : mul"
echo " Shape     : [B, 3, 4, 256, 32]  B in {${BATCHES[*]}}"
echo " Dtype     : float16"
echo " Warmup    : $WARMUP"
echo " Loops     : $LOOPS"
echo " Repeat    : $REPEAT"
echo ""
echo " Reports:"
echo "  Raw traces   : $RAW_DIR"
echo "  Parsed       : $PARSED_DIR"
echo "  Final JSON   : $FINAL_DIR/benchmark_report.json"
echo "  Final MD     : $FINAL_DIR/benchmark_report.md"
echo "============================================"
