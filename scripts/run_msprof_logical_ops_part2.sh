#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$PROJECT_ROOT"

BATCHES="1 2 4 8 16 32 64"
WARMUP=200
LOOPS=100
REPEAT=5
MSPROF="/home/developer/Ascend/cann-9.0.0/bin/msprof"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

gen_script() {
    local op=$1 impl=$2 batch=$3 type=$4
    local f="/tmp/${type}_${impl}_${op}_b${batch}.py"
    if [ "$impl" = "torch" ]; then
        local D="${PROJECT_ROOT}/operators/${op}/data"
        case "$op" in
            or)
            cat > "$f" << PYEOF
import torch, torch_npu, numpy as np; torch.npu.set_device(0)
SHAPE=[${batch},256,384]
x1=torch.from_numpy(np.fromfile("${D}/x1_b${batch}_random_mask.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
x2=torch.from_numpy(np.fromfile("${D}/x2_b${batch}_random_mask.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
for _ in range(${WARMUP}): torch.logical_or(x1,x2)
torch.npu.synchronize()
PYEOF
            ;;
            where)
            cat > "$f" << PYEOF
import torch, torch_npu, numpy as np; torch.npu.set_device(0)
SHAPE=[${batch},256,384]
cond=torch.from_numpy(np.fromfile("${D}/condition_b${batch}_bool.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
x1=torch.from_numpy(np.fromfile("${D}/x1_b${batch}_fp16.bin",dtype=np.float16).reshape(SHAPE)).npu(0)
x2=torch.from_numpy(np.fromfile("${D}/x2_b${batch}_fp16.bin",dtype=np.float16).reshape(SHAPE)).npu(0)
for _ in range(${WARMUP}): torch.where(cond.bool(),x1,x2)
torch.npu.synchronize()
PYEOF
            ;;
        esac
    elif [ "$impl" = "pypto" ]; then
        if [ "$op" = "or" ]; then
            cat > "$f" << PYEOF
import sys
sys.path.insert(0,"${PROJECT_ROOT}/operators/or/pypto/src")
sys.path.insert(0,"${PROJECT_ROOT}/operators/or/pypto/golden")
from or_impl import or_wrapper
import torch, torch_npu, numpy as np; torch.npu.set_device(0)
SHAPE=[${batch},256,384]
x1=torch.from_numpy(np.fromfile("${PROJECT_ROOT}/operators/or/data/x1_b${batch}_random_mask.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
x2=torch.from_numpy(np.fromfile("${PROJECT_ROOT}/operators/or/data/x2_b${batch}_random_mask.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
for _ in range(${WARMUP}): or_wrapper(x1,x2)
torch.npu.synchronize()
PYEOF
        fi
    fi
    echo "$f"
}

gen_msprof_script() {
    local op=$1 impl=$2 batch=$3
    local f="/tmp/msprof_${impl}_${op}_b${batch}.py"
    if [ "$impl" = "torch" ]; then
        local D="${PROJECT_ROOT}/operators/${op}/data"
        case "$op" in
            or)
            cat > "$f" << PYEOF
import torch, torch_npu, numpy as np; torch.npu.set_device(0)
SHAPE=[${batch},256,384]
x1=torch.from_numpy(np.fromfile("${D}/x1_b${batch}_random_mask.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
x2=torch.from_numpy(np.fromfile("${D}/x2_b${batch}_random_mask.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
for _ in range(${LOOPS}): torch.logical_or(x1,x2)
torch.npu.synchronize()
PYEOF
            ;;
            where)
            cat > "$f" << PYEOF
import torch, torch_npu, numpy as np; torch.npu.set_device(0)
SHAPE=[${batch},256,384]
cond=torch.from_numpy(np.fromfile("${D}/condition_b${batch}_bool.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
x1=torch.from_numpy(np.fromfile("${D}/x1_b${batch}_fp16.bin",dtype=np.float16).reshape(SHAPE)).npu(0)
x2=torch.from_numpy(np.fromfile("${D}/x2_b${batch}_fp16.bin",dtype=np.float16).reshape(SHAPE)).npu(0)
for _ in range(${LOOPS}): torch.where(cond.bool(),x1,x2)
torch.npu.synchronize()
PYEOF
            ;;
        esac
    elif [ "$impl" = "pypto" ]; then
        if [ "$op" = "or" ]; then
            cat > "$f" << PYEOF
import sys
sys.path.insert(0,"${PROJECT_ROOT}/operators/or/pypto/src")
sys.path.insert(0,"${PROJECT_ROOT}/operators/or/pypto/golden")
from or_impl import or_wrapper
import torch, torch_npu, numpy as np; torch.npu.set_device(0)
SHAPE=[${batch},256,384]
x1=torch.from_numpy(np.fromfile("${PROJECT_ROOT}/operators/or/data/x1_b${batch}_random_mask.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
x2=torch.from_numpy(np.fromfile("${PROJECT_ROOT}/operators/or/data/x2_b${batch}_random_mask.bin",dtype=np.uint8).reshape(SHAPE)).npu(0)
for _ in range(${LOOPS}): or_wrapper(x1,x2)
torch.npu.synchronize()
PYEOF
        fi
    fi
    echo "$f"
}

run_torch() {
    local op=$1 batch=$2
    local raw_dir="operators/${op}/reports/raw/torch/b${batch}"
    mkdir -p "$raw_dir"
    local warmup_script=$(gen_script "$op" "torch" "$batch" warmup)
    local msprof_script=$(gen_msprof_script "$op" "torch" "$batch")
    log "torch $op B=$batch: warmup..."
    bash scripts/run_with_npu_lock.sh "$op" "torch" "$batch" "profiler" python3 "$warmup_script" 2>&1 | tail -1
    log "torch $op B=$batch: msprof..."
    bash scripts/run_with_npu_lock.sh "$op" "torch" "$batch" "profiler" \
        $MSPROF --output="${PROJECT_ROOT}/operators/${op}/reports/raw/torch/msprof_b${batch}" --ascendcl=on --ai-core=on --task-time=l0 \
        python3 "$msprof_script" 2>&1
    local CMD_RET=$?
    local latest_prof=$(ls -td "${PROJECT_ROOT}/operators/${op}/reports/raw/torch/msprof_b${batch}/PROF_"* 2>/dev/null | head -1)
    if [ -n "$latest_prof" ] && [ -d "$latest_prof" ]; then
        cp -r "$latest_prof"/* "$raw_dir/" 2>/dev/null || true
        rm -rf "${PROJECT_ROOT}/operators/${op}/reports/raw/torch/msprof_b${batch}"
    fi
    log "torch $op B=$batch: done (ret=$CMD_RET)"
    return $CMD_RET
}

run_ascendc() {
    local op=$1 batch=$2
    local raw_dir="operators/${op}/reports/raw/ascendc/b${batch}"
    mkdir -p "$raw_dir"
    local binary="${PROJECT_ROOT}/operators/${op}/ascendc/build/${op}_ascendc"
    if [ ! -f "$binary" ]; then log "SKIP ascendc $op B=$batch"; return 1; fi
    log "ascendc $op B=$batch: msprof..."
    bash scripts/run_with_npu_lock.sh "$op" "ascendc" "$batch" "profiler" \
        $MSPROF --output="${PROJECT_ROOT}/operators/${op}/reports/raw/ascendc/msprof_b${batch}" --ascendcl=on --ai-core=on --task-time=l0 \
        "$binary" 0 "$batch" 20 8192 "$WARMUP" "$LOOPS" "$REPEAT" "${PROJECT_ROOT}/operators/${op}/data" "${PROJECT_ROOT}/operators/${op}/ascendc/build/output" 2>&1
    local CMD_RET=$?
    local latest_prof=$(ls -td "${PROJECT_ROOT}/operators/${op}/reports/raw/ascendc/msprof_b${batch}/PROF_"* 2>/dev/null | head -1)
    if [ -n "$latest_prof" ] && [ -d "$latest_prof" ]; then
        cp -r "$latest_prof"/* "$raw_dir/" 2>/dev/null || true
        rm -rf "${PROJECT_ROOT}/operators/${op}/reports/raw/ascendc/msprof_b${batch}"
    fi
    log "ascendc $op B=$batch: done (ret=$CMD_RET)"
    return $CMD_RET
}

run_pypto() {
    local op=$1 batch=$2
    if [ "$op" = "equal" ] || [ "$op" = "where" ]; then log "SKIP pypto $op B=$batch"; return 0; fi
    local raw_dir="operators/${op}/reports/raw/pypto/b${batch}"
    mkdir -p "$raw_dir"
    local warmup_script=$(gen_script "$op" "pypto" "$batch" warmup)
    local msprof_script=$(gen_msprof_script "$op" "pypto" "$batch")
    log "pypto $op B=$batch: warmup..."
    bash scripts/run_with_npu_lock.sh "$op" "pypto" "$batch" "profiler" python3 "$warmup_script" 2>&1 | tail -1
    log "pypto $op B=$batch: msprof..."
    bash scripts/run_with_npu_lock.sh "$op" "pypto" "$batch" "profiler" \
        $MSPROF --output="${PROJECT_ROOT}/operators/${op}/reports/raw/pypto/msprof_b${batch}" --ascendcl=on --ai-core=on --task-time=l0 \
        python3 "$msprof_script" 2>&1
    local CMD_RET=$?
    local latest_prof=$(ls -td "${PROJECT_ROOT}/operators/${op}/reports/raw/pypto/msprof_b${batch}/PROF_"* 2>/dev/null | head -1)
    if [ -n "$latest_prof" ] && [ -d "$latest_prof" ]; then
        cp -r "$latest_prof"/* "$raw_dir/" 2>/dev/null || true
        rm -rf "${PROJECT_ROOT}/operators/${op}/reports/raw/pypto/msprof_b${batch}"
    fi
    log "pypto $op B=$batch: done (ret=$CMD_RET)"
    return $CMD_RET
}

# === Remaining work ===

# 1) not pypto b64
log "--- Remaining: not pypto b64 ---"
run_pypto "not" "64"

# 2) or: all batches for torch, ascendc, pypto
log "##### OR: all batches #####"
for b in 1 2 4 8 16 32 64; do run_torch "or" "$b"; done
for b in 1 2 4 8 16 32 64; do run_ascendc "or" "$b"; done
for b in 1 2 4 8 16 32 64; do run_pypto "or" "$b"; done

# 3) where: all batches for torch, ascendc
log "##### WHERE: all batches #####"
for b in 1 2 4 8 16 32 64; do run_torch "where" "$b"; done
for b in 1 2 4 8 16 32 64; do run_ascendc "where" "$b"; done

log "===== PART 2 COMPLETE ====="
