#!/usr/bin/env python3
"""Build final comparison report from benchmark results."""
import os, sys, json
from datetime import datetime

REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
OPERATORS_DIR = os.path.dirname(REPORTS_DIR)

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Load all benchmark results
pypto_results = load_json(os.path.join(OPERATORS_DIR, 'pypto', 'benchmark_results.json'))
torch_results = load_json(os.path.join(OPERATORS_DIR, 'torch', 'benchmark_results.json'))

# Ascend C: parse from the raw output (already printed)
# We'll embed manually from the completed run
ascendc_data = [
    {"batch": 1, "median_us": 6.6, "mean_us": 6.7, "min_us": 6.6, "p90_us": 6.9, "std_us": 0.09, "cv": 1.38},
    {"batch": 2, "median_us": 6.6, "mean_us": 6.6, "min_us": 6.6, "p90_us": 6.8, "std_us": 0.07, "cv": 0.98},
    {"batch": 4, "median_us": 6.6, "mean_us": 6.6, "min_us": 6.5, "p90_us": 6.6, "std_us": 0.04, "cv": 0.55},
    {"batch": 8, "median_us": 8.0, "mean_us": 7.7, "min_us": 6.6, "p90_us": 8.1, "std_us": 0.58, "cv": 7.55},
    {"batch": 16, "median_us": 8.0, "mean_us": 8.0, "min_us": 7.9, "p90_us": 8.1, "std_us": 0.07, "cv": 0.89},
    {"batch": 32, "median_us": 12.6, "mean_us": 12.7, "min_us": 12.6, "p90_us": 13.0, "std_us": 0.16, "cv": 1.25},
    {"batch": 64, "median_us": 22.3, "mean_us": 22.3, "min_us": 22.2, "p90_us": 22.5, "std_us": 0.09, "cv": 0.39},
]

def get_bw(total_elements, latency_us):
    total_bytes = total_elements * 2 * 5
    return round(total_bytes / (latency_us * 1e-6) / (1024**3), 2)

batches = [1, 2, 4, 8, 16, 32, 64]
total_elems = {b: b * 256 * 384 for b in batches}

# Build comparison table
print(f"# 4-Input FP16 Add Comparison")
print(f"## Y = ((X1+X2)+X3)+X4, shape [B,256,384]")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("| B | Variant | Median (μs) | Mean (μs) | Min (μs) | P90 (μs) | Std (μs) | CV (%) | BW (GB/s) |")
print("|---|---------|-------------|-----------|----------|----------|----------|--------|-----------|")

for b in batches:
    te = total_elems[b]
    
    # PyPTO
    pr = [r for r in pypto_results['results'] if r['batch'] == b][0]
    bw = get_bw(te, pr['latency_us']['median_us'])
    print(f"| {b} | PyPTO     | {pr['latency_us']['median_us']:.1f} | {pr['latency_us']['mean_us']:.1f} | {pr['latency_us']['min_us']:.1f} | {pr['latency_us']['p90_us']:.1f} | {pr['latency_us']['std_us']:.2f} | {pr['latency_us']['cv']*100:.2f} | {bw} |")
    
    # Torch
    tr = [r for r in torch_results['results'] if r['batch'] == b][0]
    bw = get_bw(te, tr['latency_us']['median_us'])
    print(f"| {b} | Torch     | {tr['latency_us']['median_us']:.1f} | {tr['latency_us']['mean_us']:.1f} | {tr['latency_us']['min_us']:.1f} | {tr['latency_us']['p90_us']:.1f} | {tr['latency_us']['std_us']:.2f} | {tr['latency_us']['cv']*100:.2f} | {bw} |")
    
    # Ascend C
    ar = [d for d in ascendc_data if d['batch'] == b][0]
    bw = get_bw(te, ar['median_us'])
    print(f"| {b} | Ascend C  | {ar['median_us']:.1f} | {ar['mean_us']:.1f} | {ar['min_us']:.1f} | {ar['p90_us']:.1f} | {ar['std_us']:.2f} | {ar['cv']:.2f} | {bw} |")

print()
print("## Speedup vs Torch (median)")
print()
print("| B | Ascend C (μs) | Torch (μs) | Speedup |")
print("|---|---------------|------------|---------|")
for b in batches:
    tr = [r for r in torch_results['results'] if r['batch'] == b][0]
    ar = [d for d in ascendc_data if d['batch'] == b][0]
    speedup = tr['latency_us']['median_us'] / ar['median_us']
    print(f"| {b} | {ar['median_us']:.1f} | {tr['latency_us']['median_us']:.1f} | {speedup:.2f}x |")

print()
print("## Notes")
print("- **PyPTO**: 3× chained `pypto.op.add` kernels with `set_vec_tile_shapes(128, 1024)` + 2D reshape")
print("- **Torch**: Three sequential `torch.add` calls (no fusion)")
print("- **Ascend C**: Single fused kernel with 3 internal adds, tile_len=8192, block_dim=12-20")
print("- Warmup=200, loops=100, repeat=5")
print("- Device: Ascend 910B (dav-2201)")
print("- dtype: float16")
