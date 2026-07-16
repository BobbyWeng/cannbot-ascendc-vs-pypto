#!/usr/bin/env python3
"""Benchmark template for {{ operator_name }} via torch.
Fill in the op-specific code at the marked location.
"""
import os, sys, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_DIR, "common"))
from common.benchmark import compute_statistics, compute_effective_bandwidth
import torch, torch_npu

DTYPE = torch.float16
SHAPE_TAIL = {{ shape_tail | default('[12, 256, 32]') }}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def total_elems(batch):
    return batch * {{ total_elems_per_batch | default('12 * 256 * 32') }}

# ====== OP-SPECIFIC: Implement the operator call ======
def op_call(x):
    return torch.{{ operator_name | default('relu') }}(x)
# ====================================================

def bench(batch, warmup=200, loops=100, repeat=10, device=0):
    input_path = os.path.join(DATA_DIR, f"input_b{batch}_fp16.bin")
    x = torch.from_numpy(np.fromfile(input_path, dtype=np.float16).reshape([batch] + list(SHAPE_TAIL)))
    x = x.npu()
    # Warmup
    for _ in range(warmup):
        _ = op_call(x)
    torch.npu.synchronize()
    # Measure
    latencies = []
    for _ in range(repeat):
        torch.npu.synchronize()
        se = torch.npu.Event(enable_timing=True)
        ee = torch.npu.Event(enable_timing=True)
        se.record()
        for _ in range(loops):
            _ = op_call(x)
        ee.record()
        torch.npu.synchronize()
        latencies.append(se.elapsed_time(ee) * 1000.0 / loops)
    stats = compute_statistics(latencies)
    bw = compute_effective_bandwidth(
        total_elems(batch) * 2, total_elems(batch) * 2, stats["median_us"])
    return {"latencies_us": latencies, **stats, "bw_GBps": bw}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="1,2,4,8,16,32,64")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    batches = [int(b) for b in args.batch.split(",")]
    torch.npu.set_device(args.device)
    results = {}
    for batch in batches:
        print(f"Benchmarking B={batch}...")
        r = bench(batch, device=args.device)
        results[batch] = r
        print(f"  median={r['median_us']:.1f}us")
    out = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {out}")
