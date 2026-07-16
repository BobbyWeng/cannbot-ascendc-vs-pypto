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

KERNEL_TAIL = [12, 256, 256]
X2_KERNEL_TAIL = [12, 256, 1]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def total_elems(batch):
    return batch * 12 * 256 * 256


def x2_elems(batch):
    return batch * 12 * 256 * 1


def load_inputs(batch, device_id):
    x1_path = os.path.join(DATA_DIR, f"x1_b{batch}_fp16.bin")
    x2_path = os.path.join(DATA_DIR, f"x2_b{batch}_fp16.bin")
    shape = [batch] + KERNEL_TAIL
    x2_shape = [batch] + X2_KERNEL_TAIL
    x1 = torch.from_numpy(np.fromfile(x1_path, dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(x2_path, dtype=np.float16).reshape(x2_shape))
    return x1.npu(device_id), x2.npu(device_id)


def run_benchmark(batch, warmup, loops, repeat, device_id):
    torch.npu.set_device(device_id)
    x1_npu, x2_npu = load_inputs(batch, device_id)

    for _ in range(warmup):
        _ = torch.div(x1_npu, x2_npu)
    torch.npu.synchronize(device_id)

    latencies_us = []
    for _ in range(repeat):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(loops):
            _ = torch.div(x1_npu, x2_npu)
        end_event.record()
        end_event.synchronize()
        avg_us = start_event.elapsed_time(end_event) * 1000.0 / loops
        latencies_us.append(avg_us)

    stats = compute_statistics(latencies_us)

    x1_bytes = total_elems(batch) * 2
    x2_bytes = x2_elems(batch) * 2
    y_bytes = total_elems(batch) * 2
    total_bytes = x1_bytes + x2_bytes + y_bytes
    bw = compute_effective_bandwidth(total_bytes, 0, stats["median_us"])
    stats["effective_bandwidth_gbps"] = round(bw, 2)

    return {
        "operator": "div",
        "variant": "torch",
        "batch": batch,
        "logical_shape": [batch, 3, 4, 256, 256],
        "kernel_shape": [batch] + KERNEL_TAIL,
        "dtype": "float16",
        "config": {"warmup": warmup, "loops": loops, "repeat": repeat},
        "latency_us": stats,
        "minimum_logical_bytes": x1_bytes + x2_bytes + y_bytes,
        "naive_per_output_bytes": 6,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline benchmark for Div")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32",
                        help="Comma-separated batch sizes")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations")
    parser.add_argument("--loops", type=int, default=100, help="Inner loop iterations per repeat")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeat measurements")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    results = []
    for b in batches:
        result = run_benchmark(b, args.warmup, args.loops, args.repeat, args.device)
        results.append(result)
        print(json.dumps(result, indent=2))

    output = {"results": results, "framework": "torch", "operator": "div"}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
