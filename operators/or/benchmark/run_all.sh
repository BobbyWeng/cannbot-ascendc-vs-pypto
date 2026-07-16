#!/bin/bash
set -euo pipefail

OP_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
BATCHES="1,2,4,8,16,32,64"
DEVICE_ID=0
WARMUP=200
LOOPS=100
REPEAT=5

echo "=== OR Benchmark Suite ==="

echo ""
echo "--- Torch Baseline ---"
cd "$OP_DIR"
python3 torch/benchmark.py --batch "$BATCHES" --device "$DEVICE_ID" --warmup "$WARMUP" --loops "$LOOPS" --repeat "$REPEAT"

echo ""
echo "--- Ascend C (build required) ---"
echo "Skipped: requires cmake build"

echo ""
echo "--- PyPTO ---"
echo "Skipped: requires PyPTO environment"

echo ""
echo "=== Done ==="
