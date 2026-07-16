#!/bin/bash
# Benchmark runner template for {{ operator_name }}
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(dirname "$SCRIPT_DIR")"
PROJ_DIR="$(dirname "$(dirname "$OP_DIR")")"
# Source CANN
if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    source "$ASCEND_HOME_PATH/set_env.sh"
fi
# Generate data if needed
DATA_DIR="$OP_DIR/data"
if [ ! -f "$DATA_DIR/input_b1_fp16.bin" ]; then
    echo "Generating data..."
    python3 "$DATA_DIR/generation_scripts/generate_inputs.py"
    python3 "$DATA_DIR/generation_scripts/generate_reference.py"
fi
# Build ascendc if needed
if [ ! -f "$OP_DIR/ascendc/build/relu_ascendc" ]; then
    echo "Building Ascend C..."
    mkdir -p "$OP_DIR/ascendc/build"
    cmake -S "$OP_DIR/ascendc" -B "$OP_DIR/ascendc/build"
    make -C "$OP_DIR/ascendc/build" -j$(nproc)
fi
echo "Template ready. Modify for {{ operator_name }}."
