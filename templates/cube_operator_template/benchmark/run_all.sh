#!/bin/bash
# Unified benchmark runner for {{op_name}} Cube
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(dirname "$SCRIPT_DIR")"
PROJ_DIR="$(dirname "$(dirname "$OP_DIR")")"

source "$ASCEND_HOME_PATH/set_env.sh" 2>/dev/null || true

BATCHES="{{batches}}"
WARMUP=200
LOOPS=100
REPEAT=5

# === Torch ===
for b in $BATCHES; do
  msprof --output="$OP_DIR/reports/raw/torch_b${b}" \
    --ascendcl=on --ai-core=on --task-time=l0 \
    python3 "$OP_DIR/torch/benchmark.py" --batches "$b"
done

# === Ascend C ===
BUILD_DIR="$OP_DIR/ascendc/build"
for b in $BATCHES; do
  msprof --output="$OP_DIR/reports/raw/ascendc_b${b}" \
    --ascendcl=on --ai-core=on --task-time=l0 \
    "$BUILD_DIR/{{op_name}}_ascendc" 0 "$b" 20 "$WARMUP" "$LOOPS" "$REPEAT" "$OP_DIR/data" "$BUILD_DIR/output"
done

# === PyPTO (two-process) ===
for b in $BATCHES; do
  python3 -c "
import sys; sys.path.insert(0, '${OP_DIR}/pypto/src')
from {{op_name}}_impl import {{op_name}}_wrapper
import torch, torch_npu
# warmup
for _ in range(${WARMUP}): pass
torch.npu.synchronize()
"
  msprof --output="$OP_DIR/reports/raw/pypto_b${b}" \
    --ascendcl=on --ai-core=on --task-time=l0 \
    python3 -c "
import sys; sys.path.insert(0, '${OP_DIR}/pypto/src')
from {{op_name}}_impl import {{op_name}}_wrapper
import torch, torch_npu
for _ in range(${LOOPS}): pass
torch.npu.synchronize()
"
done
