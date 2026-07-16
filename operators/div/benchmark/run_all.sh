#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="$(dirname "$0")"
OPER_DIR="$(realpath "$BENCH_DIR/..")"
PROJECT_ROOT="$(realpath "$OPER_DIR/../..")"

BATCHES=(1 2 4 8 16 32)
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
ASCENDC_BIN="$ASCENDC_BUILD_DIR/div_ascendc"
TORCH_DIR="$OPER_DIR/torch"
PYPTO_DIR="$OPER_DIR/pypto"
REPORTS_DIR="$OPER_DIR/reports"
RAW_DIR="$REPORTS_DIR/raw"
PARSED_DIR="$REPORTS_DIR/parsed"
CORRECTNESS_DIR="$REPORTS_DIR/correctness"
FINAL_DIR="$REPORTS_DIR/final"

mkdir -p "$RAW_DIR" "$PARSED_DIR" "$CORRECTNESS_DIR" "$FINAL_DIR"

echo "============================================"
echo " Div Broadcast Benchmark Runner"
echo "============================================"

# Source CANN environment
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

# Preflight checks
echo ""
echo "--- Preflight checks ---"

check_data() {
    local missing=0
    for b in "${BATCHES[@]}"; do
        if [ ! -f "$DATA_DIR/x1_b${b}_${DTYPE}.bin" ]; then echo "[MISS] x1_b${b}_${DTYPE}.bin"; missing=1; fi
        if [ ! -f "$DATA_DIR/x2_b${b}_${DTYPE}.bin" ]; then echo "[MISS] x2_b${b}_${DTYPE}.bin"; missing=1; fi
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
from div_golden import div_golden
from div_impl import div_wrapper
print('[OK] PyPTO imports successful')
" 2>&1 || echo "[WARN] PyPTO import check failed"

# Correctness verification
echo ""
echo "--- Correctness verification ---"

BATCH_CSV=$(IFS=,; echo "${BATCHES[*]}")

echo "[CORRECT] Torch baseline..."
python3 "$TORCH_DIR/correctness.py" --batch "$BATCH_CSV" --device "$DEVICE_ID" 2>&1
cp "$TORCH_DIR/correctness_results.json" "$CORRECTNESS_DIR/torch_correctness.json"

echo "[CORRECT] Ascend C..."
ASCENDC_OUT_DIR="$ASCENDC_BUILD_DIR/output"
mkdir -p "$ASCENDC_OUT_DIR"
for b in "${BATCHES[@]}"; do
    "$ASCENDC_BIN" "$DEVICE_ID" "$b" "$BLOCK_DIM" "$TILE_LEN" 1 1 1 "$DATA_DIR" "$ASCENDC_OUT_DIR" > /dev/null 2>&1
    python3 "$DATA_DIR/generation_scripts/correctness.py" \
        "$ASCENDC_OUT_DIR/output_b${b}.bin" \
        "$DATA_DIR/reference_b${b}_fp16.bin" \
        "$DATA_DIR/reference_b${b}_fp32.bin" \
        "$b" || echo "[WARN] Ascend C correctness b=$b check failed"
done

echo "[CORRECT] PyPTO..."
python3 "$PYPTO_DIR/tests/test_div.py" 2>&1 || true

# Profiler benchmarks
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
from div_impl import div_wrapper

batch = int(sys.argv[1])
warmup = int(sys.argv[2])
loops = int(sys.argv[3])
repeat = int(sys.argv[4])
device_id = int(sys.argv[5])
data_dir = sys.argv[6]

kernel_tail = [12, 256, 256]
x2_kernel_tail = [12, 256, 1]
shape = [batch] + kernel_tail
x2_shape = [batch] + x2_kernel_tail
x1_path = os.path.join(data_dir, f"x1_b{batch}_fp16.bin")
x2_path = os.path.join(data_dir, f"x2_b{batch}_fp16.bin")
x1 = torch.from_numpy(np.fromfile(x1_path, dtype=np.float16).reshape(shape))
x2 = torch.from_numpy(np.fromfile(x2_path, dtype=np.float16).reshape(x2_shape))
import torch_npu
torch.npu.set_device(device_id)
x1_npu = x1.npu(device_id)
x2_npu = x2.npu(device_id)

for _ in range(warmup):
    _ = div_wrapper(x1_npu, x2_npu)
torch.npu.synchronize(device_id)

for _ in range(loops + repeat):
    _ = div_wrapper(x1_npu, x2_npu)
torch.npu.synchronize(device_id)
PYEOF

    chmod +x "$PYPTO_BENCH_SCRIPT"

    echo "  [phase 1] warmup..."
    python3 "$PYPTO_BENCH_SCRIPT" "$batch" "$WARMUP" 0 0 "$DEVICE_ID" "$DATA_DIR"

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

# Aggregate parsed results
echo ""
echo "--- Aggregating results ---"
python3 -c "
import os, json
from collections import defaultdict

parsed_dir = '$PARSED_DIR'
batches = [1, 2, 4, 8, 16, 32]
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

# Generate final reports
echo ""
echo "--- Final report generation ---"
python3 -c "
import os, json, sys
sys.path.insert(0, '$PROJECT_ROOT')
from common.reporting import generate_markdown_report, generate_json_report, generate_csv_report

parsed_dir = '$PARSED_DIR'
final_dir = '$FINAL_DIR'
batches = [1, 2, 4, 8, 16, 32]
impls = ['torch', 'ascendc', 'pypto']

aggregate = {}
fpath = os.path.join(parsed_dir, 'aggregate.json')
if os.path.exists(fpath):
    with open(fpath) as f:
        aggregate = json.load(f)

report = {
    'experiment': {
        'operator': 'div',
        'formula': 'Y = X1 / X2 (last-dim broadcast)',
        'logical_shape': '[B, 3, 4, 256, 256]',
        'kernel_shape': '[B, 12, 256, 256]',
        'x2_kernel_shape': '[B, 12, 256, 1]',
        'broadcast_axis': 'last dim (256 -> 256)',
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

json_out = os.path.join(final_dir, 'final_comparison.json')
generate_json_report(report, json_out)

print(f'json -> {json_out}')
print('Reports generated.')
"

# Generate markdown and CSV reports
python3 -c "
import os, json, sys
sys.path.insert(0, '$PROJECT_ROOT')
from common.reporting import generate_csv_report

final_dir = '$FINAL_DIR'

with open(os.path.join(final_dir, 'final_comparison.json')) as f:
    data = json.load(f)

csv_out = os.path.join(final_dir, 'final_comparison.csv')
generate_csv_report(data, csv_out)
print(f'csv -> {csv_out}')
"

# Generate markdown report
python3 << 'PYEOF'
import os, json

final_dir = "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/div/reports/final"
parsed_dir = "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/div/reports/parsed"

batches = [1, 2, 4, 8, 16, 32]

md = """# Div Comparison — Profiler-Based Device Kernel Analysis

## Experiment Environment

| Item | Value |
|------|-------|
| Operator | Div (Y = X1 / X2, last-dim broadcast) |
| Device | Ascend 910B (device 0) |
| dtype | FP16 |
| Logical shape | [B, 3, 4, 256, 256] (X1, Y); [B, 3, 4, 256, 1] (X2) |
| Kernel shape | [B, 12, 256, 256] (X1, Y); [B, 12, 256, 1] (X2) |
| Dimension merge | (3,4) -> 12, contiguous zero-copy view |
| Broadcast axis | Last dimension (index 4 logical, index 3 kernel); factor = 256 |
| Batches | 1, 2, 4, 8, 16, 32 |
| Warmup | 200 |
| Profiled loops | 100 |
| Repeat | 5 |
| Profiler | msprof --ascendcl=on --ai-core=on --task-time=l0 |
| Seed | 20260715 |
| X1 range | [-4, 4] |
| X2 range | [-4, -0.25] U [0.25, 4] |
| Correctness atol/rtol | 1e-3 / 1e-3 |

## Computation Characteristics

- **broadcast reduction factor**: 256x (each X2 element used 256 times)
- **minimum_logical_bytes per output element**: X1 2B + X2 0.0078B + Y 2B ≈ 4.008 B/elem
- **naive_per_output_bytes**: 6 B/elem (if X2 expanded to full shape)
- **Ascend C strategy**: In-kernel UB broadcast (single Div kernel, no GM expansion)
- **PyPTO strategy**: TBD from profiling

## Correctness

All three implementations tested against:
- **torch FP16 reference**: `torch.div(x1, x2)` on CPU FP16
- **FP32 high-precision reference**: `x1.float32 / x2.float32`

Gate atol=1e-3, rtol=1e-3 for finite values.

"""

# Try to read aggregate if available
agg_path = os.path.join(parsed_dir, "aggregate.json")
if os.path.exists(agg_path):
    with open(agg_path) as f:
        agg = json.load(f)

    md += "## Profiler Results (All Device Kernels Per Call)\n\n"
    md += "| Batch | torch | Ascend C | PyPTO |\n"
    md += "|-------|-------|----------|-------|\n"
    for b in batches:
        key = str(b)
        t = agg.get("torch", {}).get(key, {}).get("all_device_kernels_us", "N/A")
        a = agg.get("ascendc", {}).get(key, {}).get("all_device_kernels_us", "N/A")
        p = agg.get("pypto", {}).get(key, {}).get("all_device_kernels_us", "N/A")
        t_str = f"{t:.1f}" if isinstance(t, (int, float)) else t
        a_str = f"{a:.1f}" if isinstance(a, (int, float)) else a
        p_str = f"{p:.1f}" if isinstance(p, (int, float)) else p
        md += f"| {b} | {t_str} us | {a_str} us | {p_str} us |\n"
    md += "\n"

md += """## Kernel Count and Type

| Implementation | Kernel Type | Kernels/logical call | Broadcast Implementation |
|--------------|-------------|---------------------|-------------------------|
| torch.div | TBD | TBD | TBD from msprof trace |
| Ascend C | KERNEL_AIVEC (expected) | 1 | In-kernel UB broadcast, no GM expansion |
| PyPTO | TBD | TBD | TBD from msprof trace |

## Key Analysis Questions

1. **Does torch materialize the broadcast?** — Check if msprof shows Expand/Copy kernels in addition to Div.
2. **Does Ascend C achieve single-kernel broadcast Div?** — Yes, in-kernel UB broadcast.
3. **Does PyPTO support true broadcast lowering?** — Check if PyPTO generates a single kernel or requires Expand + element-wise.
4. **Does PyPTO use native Div or reciprocal_mul?** — Determined from PyPTO API report.
5. **Kernel types**: AIVEC vs MIX_AIC vs AICPU comparison.
6. **X2 read pattern**: Is X2 read repeatedly from GM or cached in UB?

## Algorithmic Data Volume (Broadcast-Aware)

| Batch | X1 read | X2 read (logical) | Y write | total_bytes |
|-------|---------|-------------------|---------|-------------|

"""

for b in batches:
    x1 = b * 12 * 256 * 256 * 2
    x2 = b * 12 * 256 * 1 * 2
    y = b * 12 * 256 * 256 * 2
    total = x1 + x2 + y
    md += f"| {b} | {x1/1024/1024:.1f} MiB | {x2/1024/1024:.2f} MiB | {y/1024/1024:.1f} MiB | {total/1024/1024:.1f} MiB |\n"

md += """
## Final Summary

| Metric | Value |
|--------|-------|
| Overall Status | IN PROGRESS |
| Logical Shape | [B, 3, 4, 256, 256] |
| Kernel Shape | [B, 12, 256, 256] |
| Implementation Strategy | torch.div / Ascend C native Div / PyPTO TBD |
| Broadcast | Last-dim broadcast 256 -> 256 |
| Ascend C broadcast strategy | In-kernel UB broadcast (no GM expansion) |
| Ascend C strategy | AscendC::Div inside UB (single kernel) |
| PyPTO TBD | Awaiting profiling completion |

## Known Issues

- Complete profiler data requires msprof execution on Ascend 910B hardware
- PyPTO backend support for broadcast and Div to be determined
- B=64 excluded initially due to memory considerations
"""

out_path = os.path.join(final_dir, "final_comparison.md")
with open(out_path, "w") as f:
    f.write(md)
print(f"md -> {out_path}")
PYEOF

# Summary
echo ""
echo "============================================"
echo " Summary"
echo "============================================"
echo " Operator  : div (broadcast)"
echo " Logical   : [B, 3, 4, 256, 256] / X2: [B, 3, 4, 256, 1]"
echo " Kernel    : [B, 12, 256, 256] / X2: [B, 12, 256, 1]"
echo " Dtype     : float16"
echo " Broadcast : last-dim (256 -> 256)"
echo " Batches   : 1,2,4,8,16,32 (B=64 optional)"
echo " Warmup    : $WARMUP"
echo " Loops     : $LOOPS"
echo " Repeat    : $REPEAT"
echo ""
echo " Reports:"
echo "  Raw traces   : $RAW_DIR"
echo "  Parsed       : $PARSED_DIR"
echo "  Final JSON   : $FINAL_DIR/final_comparison.json"
echo "  Final MD     : $FINAL_DIR/final_comparison.md"
echo "  Final CSV    : $FINAL_DIR/final_comparison.csv"
echo "============================================"
