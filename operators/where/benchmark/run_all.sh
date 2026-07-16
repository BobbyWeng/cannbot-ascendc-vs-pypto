#!/bin/bash
# Unified benchmark runner for Where
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(dirname "$SCRIPT_DIR")"
PROJ_DIR="$(dirname "$(dirname "$OP_DIR")")"

source "$ASCEND_HOME_PATH/set_env.sh" 2>/dev/null || true

DATA_DIR="$OP_DIR/data"
REPORT_RAW="$OP_DIR/reports/raw"
REPORT_PARSED="$OP_DIR/reports/parsed"
mkdir -p "$REPORT_RAW" "$REPORT_PARSED"

BATCHES="1 2 4 8 16 32 64"
WARMUP=200
LOOPS=100
REPEAT=5

# --- Torch ---
echo "=== Torch benchmark ==="
for b in $BATCHES; do
  echo "  B=${b}..."
  msprof --output="$REPORT_RAW/torch_b${b}" \
    --ascendcl=on --ai-core=on --task-time=l0 \
    python3 "$OP_DIR/torch/benchmark.py" \
      --batch "$b" --warmup "$WARMUP" --loops "$LOOPS" --repeat "$REPEAT"
  python3 "$OP_DIR/benchmark/parse_profiler.py" \
    "$REPORT_RAW/torch_b${b}" "$REPORT_PARSED/torch_b${b}.json"
done

# --- Ascend C ---
echo "=== Ascend C benchmark ==="
BUILD_DIR="$OP_DIR/ascendc/build"
if [ ! -f "$BUILD_DIR/where_ascendc" ]; then
  echo "Building Ascend C..."
  mkdir -p "$BUILD_DIR"
  cmake -S "$OP_DIR/ascendc" -B "$BUILD_DIR"
  cmake --build "$BUILD_DIR" -j
fi
for b in $BATCHES; do
  echo "  B=${b}..."
  msprof --output="$REPORT_RAW/ascendc_b${b}" \
    --ascendcl=on --ai-core=on --task-time=l0 \
    "$BUILD_DIR/where_ascendc" 0 "$b" 20 8192 "$WARMUP" "$LOOPS" "$REPEAT" "$DATA_DIR" "$BUILD_DIR/output"
  python3 "$OP_DIR/benchmark/parse_profiler.py" \
    "$REPORT_RAW/ascendc_b${b}" "$REPORT_PARSED/ascendc_b${b}.json"
done

# --- PyPTO (two-process) ---
echo "=== PyPTO benchmark ==="
for b in $BATCHES; do
  echo "  B=${b}..."
  # Warmup (no profiler)
  python3 -c "
import sys; sys.path.insert(0, '${OP_DIR}/pypto/src')
from where_impl import where_wrapper
import torch, torch_npu
cond=torch.randint(0,2,(${b},256,384),dtype=torch.uint8).npu()
x1=torch.randn(${b},256,384,dtype=torch.float16).npu()
x2=torch.randn(${b},256,384,dtype=torch.float16).npu()
for _ in range(${WARMUP}): where_wrapper(cond,x1,x2)
torch.npu.synchronize()
"
  # Measure
  msprof --output="$REPORT_RAW/pypto_b${b}" \
    --ascendcl=on --ai-core=on --task-time=l0 \
    python3 -c "
import sys; sys.path.insert(0, '${OP_DIR}/pypto/src')
from where_impl import where_wrapper
import torch, torch_npu
cond=torch.randint(0,2,(${b},256,384),dtype=torch.uint8).npu()
x1=torch.randn(${b},256,384,dtype=torch.float16).npu()
x2=torch.randn(${b},256,384,dtype=torch.float16).npu()
for _ in range(${LOOPS}): where_wrapper(cond,x1,x2)
torch.npu.synchronize()
"
  python3 "$OP_DIR/benchmark/parse_profiler.py" \
    "$REPORT_RAW/pypto_b${b}" "$REPORT_PARSED/pypto_b${b}.json"
done

echo "=== All benchmarks complete ==="
echo "Raw data: $REPORT_RAW"
echo "Parsed data: $REPORT_PARSED"
