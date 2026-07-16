#!/bin/bash
# Run verified JIT template test
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Verified JIT Template Test ==="
echo "Project root: $PROJECT_ROOT"
echo "cwd: $(pwd)"
echo "Python: $(which python3)"
echo "PYTHONPATH: ${PYTHONPATH:-<unset>}"
echo ""

echo "=== Step 1: Source introspection ==="
python3 common/pypto/verified_jit_template/scripts/inspect_source.py \
    --module-path "common/pypto/verified_jit_template/src" \
    --module-name "relu_impl" \
    --func-name "relu_kernel_2d" \
    --output "/tmp/verified_jit_source_check.json"

echo ""
echo "=== Step 2: Run NPU correctness ==="
# Check NPU lock
bash scripts/acquire_npu_lock.sh "verified_jit_template" "pypto" "1" "test" 0
python3 common/pypto/verified_jit_template/tests/test_relu.py
bash scripts/release_npu_lock.sh "verified_jit_template"
