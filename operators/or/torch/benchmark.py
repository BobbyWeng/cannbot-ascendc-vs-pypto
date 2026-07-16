#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_DIR)
import torch
import torch_npu
from common.benchmark.benchmark_utils import compute_statistics, compute_effective_bandwidth

SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def total_elems(batch):
    return batch * 256 * 384


def load_inputs(batch, device_id):
    shape = [batch] + SHAPE_TAIL
    x1 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x1_b{batch}_random_mask.bin"), dtype=np.uint8).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x2_b{batch}_random_mask.bin"), dtype=np.uint8).reshape(shape))
    return x1.npu(device_id), x2.npu(device_id)


def logical_or_op(x1, x2):
    return torch.logical_or(x1, x2).to(torch.uint8)


def run_benchmark(batch, warmup, loops, repeat, device_id):
    torch.npu.set_device(device_id)
    x1_npu, x2_npu = load_inputs(batch, device_id)

    for _ in range(warmup):
        _ = logical_or_op(x1_npu, x2_npu)
    torch.npu.synchronize(device_id)

    latencies_us = []
    for _ in range(repeat):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(loops):
            _ = logical_or_op(x1_npu, x2_npu)
        end_event.record()
        end_event.synchronize()
        avg_us = start_event.elapsed_time(end_event) * 1000.0 / loops
        latencies_us.append(avg_us)

    stats = compute_statistics(latencies_us)

    elem_bytes = total_elems(batch)
    input_bytes = elem_bytes * 2
    output_bytes = elem_bytes
    total_bytes = input_bytes + output_bytes
    bw = compute_effective_bandwidth(total_bytes, 0, stats["median_us"])
    stats["effective_bandwidth_gbps"] = round(bw, 2)

    return {
        "operator": "or",
        "variant": "torch",
        "batch": batch,
        "shape": [batch] + SHAPE_TAIL,
        "dtype": "bool",
        "config": {
            "warmup": warmup,
            "loops": loops,
            "repeat": repeat,
        },
        "latency_us": stats,
        "bytes_per_elem": 3,
        "total_bytes": total_elems(batch) * 3,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline benchmark for OR")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    results = []
    for b in batches:
        result = run_benchmark(b, args.warmup, args.loops, args.repeat, args.device)
        results.append(result)
        print(json.dumps(result, indent=2))

    output = {"results": results, "framework": "torch", "operator": "or"}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
