#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_DIR)
import torch
import torch_npu
from common.benchmark import compute_statistics, compute_effective_bandwidth

SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def total_elems(batch):
    return batch * 256 * 384


def run_benchmark(batch, case_name, warmup, loops, repeat, device_id):
    torch.npu.set_device(device_id)
    shape = [batch] + SHAPE_TAIL
    x_path = os.path.join(DATA_DIR, f"x_b{batch}_{case_name}.bin")
    x = torch.from_numpy(np.fromfile(x_path, dtype=np.uint8).reshape(shape))
    x_npu = x.npu(device_id)

    for _ in range(warmup):
        _ = torch.logical_not(x_npu.bool()).byte()
    torch.npu.synchronize(device_id)

    latencies_us = []
    for _ in range(repeat):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(loops):
            _ = torch.logical_not(x_npu.bool()).byte()
        end_event.record()
        end_event.synchronize()
        avg_us = start_event.elapsed_time(end_event) * 1000.0 / loops
        latencies_us.append(avg_us)

    stats = compute_statistics(latencies_us)

    elem_bytes = total_elems(batch) * 1
    total_bytes = elem_bytes * 2

    return {
        "operator": "not",
        "variant": "torch",
        "batch": batch,
        "case": case_name,
        "shape": [batch] + SHAPE_TAIL,
        "dtype": "bool (uint8)",
        "config": {
            "warmup": warmup,
            "loops": loops,
            "repeat": repeat,
        },
        "latency_us": stats,
        "bytes_per_elem": 2,
        "total_bytes": total_elems(batch) * 1 * 2,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline benchmark for Not")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes")
    parser.add_argument("--case", type=str, default="random_mask",
                        help="Boundary case name")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations")
    parser.add_argument("--loops", type=int, default=100, help="Inner loop iterations per repeat")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeat measurements")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    results = []
    for b in batches:
        result = run_benchmark(b, args.case, args.warmup, args.loops, args.repeat, args.device)
        results.append(result)
        print(json.dumps(result, indent=2))

    output = {"results": results, "framework": "torch", "operator": "not"}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
