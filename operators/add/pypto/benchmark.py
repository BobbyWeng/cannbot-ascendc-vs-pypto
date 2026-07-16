#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import torch
import torch_npu
from common.benchmark import compute_statistics, compute_effective_bandwidth
from add_impl import add_4

SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def total_elems(batch):
    return batch * 256 * 384


def load_input(batch, device_id, name):
    path = os.path.join(DATA_DIR, f"{name}_b{batch}_fp16.bin")
    shape = [batch] + SHAPE_TAIL
    x = torch.from_numpy(np.fromfile(path, dtype=np.float16).reshape(shape))
    return x.npu(device_id)


def run_benchmark(batch, warmup, loops, repeat, device_id):
    torch.npu.set_device(device_id)
    x1 = load_input(batch, device_id, "x1")
    x2 = load_input(batch, device_id, "x2")
    x3 = load_input(batch, device_id, "x3")
    x4 = load_input(batch, device_id, "x4")

    for _ in range(warmup):
        _ = add_4(x1, x2, x3, x4)
    torch.npu.synchronize(device_id)

    latencies_us = []
    for _ in range(repeat):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(loops):
            _ = add_4(x1, x2, x3, x4)
        end_event.record()
        end_event.synchronize()
        avg_us = start_event.elapsed_time(end_event) * 1000.0 / loops
        latencies_us.append(avg_us)

    stats = compute_statistics(latencies_us)

    elem_bytes = total_elems(batch) * 2
    input_bytes = elem_bytes * 4
    output_bytes = elem_bytes
    bw = compute_effective_bandwidth(input_bytes, output_bytes, stats["median_us"])
    stats["effective_bandwidth_gbps"] = round(bw, 2)

    return {
        "operator": "add",
        "variant": "pypto",
        "batch": batch,
        "shape": [batch] + SHAPE_TAIL,
        "dtype": "float16",
        "config": {"warmup": warmup, "loops": loops, "repeat": repeat},
        "latency_us": stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyPTO benchmark for 4-input Add")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    results = []
    for b in batches:
        result = run_benchmark(b, args.warmup, args.loops, args.repeat, args.device)
        results.append(result)
        print(json.dumps(result, indent=2))

    output = {"results": results, "framework": "pypto", "operator": "add"}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
