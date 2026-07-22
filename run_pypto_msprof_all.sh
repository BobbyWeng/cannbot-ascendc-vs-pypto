#!/usr/bin/env bash
set -euo pipefail

source ~/Ascend/ascend-toolkit/set_env.sh || echo "[WARN] CANN env not sourced"

PROJECT_ROOT="$(dirname "$0")"
cd "$PROJECT_ROOT"

BATCH=1
WARMUP=200
LOOPS=100

# Copy parse_profiler.py to any operator that is missing it
for op in div equal where transpose reduce_sum matmul softmax; do
    mkdir -p "operators/$op/benchmark"
    if [ ! -f "operators/$op/benchmark/parse_profiler.py" ]; then
        cp operators/relu/benchmark/parse_profiler.py "operators/$op/benchmark/parse_profiler.py"
        echo "[SETUP] Copied parse_profiler.py to $op/benchmark/"
    fi
done

RESULTS_FILE="reports/runtime/pypto_msprof_results.json"

echo "{" > "$RESULTS_FILE"

first=true

for op in div equal where transpose reduce_sum matmul softmax; do
    echo ""
    echo "=========================================="
    echo " Operator: $op"
    echo "=========================================="

    OP_DIR="operators/$op"
    RAW_DIR="$OP_DIR/reports/raw"
    PARSED_DIR="$OP_DIR/reports/parsed"
    mkdir -p "$RAW_DIR" "$PARSED_DIR"

    # ---- Step 1: Correctness check ----
    echo "--- Correctness check ---"
    pushd "$OP_DIR/pypto" > /dev/null

    if [ "$op" = "matmul" ]; then
        CORRECT_CMD="python3 test_matmul.py --test smoke 2>&1"
    else
        CORRECT_CMD="python3 tests/test_${op}.py 2>&1"
    fi

    if ! CORRECT_OUTPUT=$(eval "$CORRECT_CMD"); then
        echo "$CORRECT_OUTPUT"
        echo "[SKIP] $op: correctness failed, skipping msprof"
        if [ "$first" = true ]; then first=false; else echo "," >> "$RESULTS_FILE"; fi
        cat >> "$RESULTS_FILE" << JSONBLOCK
  "$op": {"status": "SKIP", "reason": "Correctness FAIL", "detail": $(echo "$CORRECT_OUTPUT" | head -20 | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")}
JSONBLOCK
        popd > /dev/null
        continue
    fi
    echo "$CORRECT_OUTPUT"
    echo "[PASS] $op: correctness passed"
    popd > /dev/null

    # ---- Step 2: Two-process msprof ----
    echo "--- msprof measurement ---"

    PYPTO_SCRIPT=$(mktemp)

    case "$op" in
        div)
            cat > "$PYPTO_SCRIPT" << 'PYEOF'
import os, sys, torch, numpy as np
import torch_npu
sys.path.insert(0, 'operators/div/pypto/golden')
sys.path.insert(0, 'operators/div/pypto/src')
from div_impl import div_wrapper
B = int(sys.argv[1]); warmup = int(sys.argv[2]); loops = int(sys.argv[3])
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
torch.npu.set_device(DEVICE_ID)
shape = (B, 12, 256, 256); x2_shape = (B, 12, 256, 1)
torch.manual_seed(20260715)
x1 = torch.randn(shape, dtype=torch.float16) * 2.0
x2_sign = torch.sign(torch.randn(x2_shape, dtype=torch.float16))
x2_mag = torch.rand(x2_shape, dtype=torch.float16) * 3.75 + 0.25
x2 = (x2_sign * x2_mag).clamp(-4.0, -0.25) + (x2_sign * x2_mag).clamp(0.25, 4.0)
x1_npu = x1.npu(DEVICE_ID); x2_npu = x2.npu(DEVICE_ID)
for _ in range(warmup): div_wrapper(x1_npu, x2_npu)
torch.npu.synchronize(DEVICE_ID)
for _ in range(loops): div_wrapper(x1_npu, x2_npu)
torch.npu.synchronize(DEVICE_ID)
PYEOF
;;
        equal)
            cat > "$PYPTO_SCRIPT" << 'PYEOF'
import os, sys, torch, numpy as np
import torch_npu
sys.path.insert(0, 'operators/equal/pypto/golden')
sys.path.insert(0, 'operators/equal/pypto/src')
from equal_impl import equal_wrapper
B = int(sys.argv[1]); warmup = int(sys.argv[2]); loops = int(sys.argv[3])
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
torch.npu.set_device(DEVICE_ID)
shape = (B, 256, 384)
torch.manual_seed(20260715)
x1 = torch.randn(shape, dtype=torch.float16); x2 = torch.randn(shape, dtype=torch.float16)
x1_npu = x1.npu(DEVICE_ID); x2_npu = x2.npu(DEVICE_ID)
for _ in range(warmup): equal_wrapper(x1_npu, x2_npu)
torch.npu.synchronize(DEVICE_ID)
for _ in range(loops): equal_wrapper(x1_npu, x2_npu)
torch.npu.synchronize(DEVICE_ID)
PYEOF
;;
        where)
            cat > "$PYPTO_SCRIPT" << 'PYEOF'
import os, sys, torch, numpy as np
import torch_npu
sys.path.insert(0, 'operators/where/pypto/golden')
sys.path.insert(0, 'operators/where/pypto/src')
from where_impl import where_wrapper
B = int(sys.argv[1]); warmup = int(sys.argv[2]); loops = int(sys.argv[3])
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
torch.npu.set_device(DEVICE_ID)
shape = (B, 256, 384)
torch.manual_seed(20260715)
condition = torch.randn(shape, dtype=torch.float16) > 0
x1 = torch.randn(shape, dtype=torch.float16); x2 = torch.randn(shape, dtype=torch.float16)
cond_npu = condition.npu(DEVICE_ID); x1_npu = x1.npu(DEVICE_ID); x2_npu = x2.npu(DEVICE_ID)
for _ in range(warmup): where_wrapper(cond_npu, x1_npu, x2_npu)
torch.npu.synchronize(DEVICE_ID)
for _ in range(loops): where_wrapper(cond_npu, x1_npu, x2_npu)
torch.npu.synchronize(DEVICE_ID)
PYEOF
;;
        transpose)
            cat > "$PYPTO_SCRIPT" << 'PYEOF'
import os, sys, torch, numpy as np
import torch_npu
sys.path.insert(0, 'operators/transpose/pypto/golden')
sys.path.insert(0, 'operators/transpose/pypto/src')
from transpose_impl import transpose_wrapper
B = int(sys.argv[1]); warmup = int(sys.argv[2]); loops = int(sys.argv[3])
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
torch.npu.set_device(DEVICE_ID)
H, W = 256, 384
torch.manual_seed(20260715)
x = torch.randn(B, H, W, dtype=torch.float16)
x_npu = x.npu(DEVICE_ID)
for _ in range(warmup): transpose_wrapper(x_npu)
torch.npu.synchronize(DEVICE_ID)
for _ in range(loops): transpose_wrapper(x_npu)
torch.npu.synchronize(DEVICE_ID)
PYEOF
;;
        reduce_sum)
            cat > "$PYPTO_SCRIPT" << 'PYEOF'
import os, sys, torch, numpy as np
import torch_npu
sys.path.insert(0, 'operators/reduce_sum/pypto/src')
from reduce_sum_impl import reduce_sum_wrapper
B = int(sys.argv[1]); warmup = int(sys.argv[2]); loops = int(sys.argv[3])
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
torch.npu.set_device(DEVICE_ID)
shape = (B, 256, 384)
torch.manual_seed(20260715)
x = torch.randn(shape, dtype=torch.float16)
x_npu = x.npu(DEVICE_ID)
for _ in range(warmup): reduce_sum_wrapper(x_npu)
torch.npu.synchronize(DEVICE_ID)
for _ in range(loops): reduce_sum_wrapper(x_npu)
torch.npu.synchronize(DEVICE_ID)
PYEOF
;;
        matmul)
            cat > "$PYPTO_SCRIPT" << 'PYEOF'
import os, sys, torch, numpy as np
import torch_npu
sys.path.insert(0, 'operators/matmul/pypto')
from matmul_impl import matmul_wrapper
B = int(sys.argv[1]); warmup = int(sys.argv[2]); loops = int(sys.argv[3])
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
torch.npu.set_device(DEVICE_ID)
shape_a = (B, 12, 256, 256); shape_b = (B, 12, 256, 32)
torch.manual_seed(20260715)
A = torch.randn(shape_a, dtype=torch.float16); B_mat = torch.randn(shape_b, dtype=torch.float16)
A_npu = A.npu(DEVICE_ID); B_npu = B_mat.npu(DEVICE_ID)
for _ in range(warmup): matmul_wrapper(A_npu, B_npu)
torch.npu.synchronize(DEVICE_ID)
for _ in range(loops): matmul_wrapper(A_npu, B_npu)
torch.npu.synchronize(DEVICE_ID)
PYEOF
;;
        softmax)
            cat > "$PYPTO_SCRIPT" << 'PYEOF'
import os, sys, torch, numpy as np
import torch_npu
sys.path.insert(0, 'operators/softmax/pypto/golden')
sys.path.insert(0, 'operators/softmax/pypto/src')
from softmax_impl import softmax_wrapper
B = int(sys.argv[1]); warmup = int(sys.argv[2]); loops = int(sys.argv[3])
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
torch.npu.set_device(DEVICE_ID)
shape = (B, 256, 32)
torch.manual_seed(20260715)
x = torch.randn(shape, dtype=torch.float16)
x_npu = x.npu(DEVICE_ID)
for _ in range(warmup): softmax_wrapper(x_npu)
torch.npu.synchronize(DEVICE_ID)
for _ in range(loops): softmax_wrapper(x_npu)
torch.npu.synchronize(DEVICE_ID)
PYEOF
;;
    esac

    chmod +x "$PYPTO_SCRIPT"
    LABEL="pypto_b${BATCH}"
    OUT_DIR="$RAW_DIR/${LABEL}"
    PARSED="$PARSED_DIR/${LABEL}.json"

    echo "  [phase 1] warmup..."
    python3 "$PYPTO_SCRIPT" "$BATCH" "$WARMUP" 0 2>&1

    echo "  [phase 2] msprof measurement..."
    msprof --output="$OUT_DIR" --ascendcl=on --ai-core=on --task-time=l0 \
        python3 "$PYPTO_SCRIPT" "$BATCH" 0 "$LOOPS" 2>&1

    python3 "$OP_DIR/benchmark/parse_profiler.py" "$OUT_DIR" "$PARSED" 2>&1

    rm -f "$PYPTO_SCRIPT"

    # Read parsed result
    if [ -f "$PARSED" ]; then
        RESULT_JSON=$(python3 -c "import json; d=json.load(open('$PARSED')); print(json.dumps(d))")
        echo "[OK] $op: msprof data saved to $PARSED"
    else
        RESULT_JSON='{"error": "parsing failed"}'
        echo "[FAIL] $op: parsing failed"
    fi

    if [ "$first" = true ]; then first=false; else echo "," >> "$RESULTS_FILE"; fi
    cat >> "$RESULTS_FILE" << JSONBLOCK
  "$op": {"status": "DONE", "parsed": $RESULT_JSON}
JSONBLOCK

done

echo "" >> "$RESULTS_FILE"
echo "}" >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo " All done! Results in $RESULTS_FILE"
echo "=========================================="

python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
for op, info in data.items():
    status = info.get('status', '?')
    if status == 'DONE':
        p = info.get('parsed', {})
        print(f'{op:12s} | DONE | primary={p.get(\"primary_compute_kernel_us\",\"?\")}us | count={p.get(\"kernel_count\",\"?\")}')
    else:
        print(f'{op:12s} | {status} | {info.get(\"reason\",\"\")}')
"
