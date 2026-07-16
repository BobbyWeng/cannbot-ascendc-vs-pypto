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


def load_inputs(batch, device_id):
    shape = [batch] + SHAPE_TAIL
    cond = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"condition_b{batch}_bool.bin"), dtype=np.uint8).reshape(shape))
    x1 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x1_b{batch}_fp16.bin"), dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x2_b{batch}_fp16.bin"), dtype=np.float16).reshape(shape))
    return cond.npu(device_id), x1.npu(device_id), x2.npu(device_id)


def run_where(cond, x1, x2):
    return torch.where(cond.bool(), x1, x2)


def run_benchmark(batch, warmup, loops, repeat, device_id):
    torch.npu.set_device(device_id)
    cond_npu, x1_npu, x2_npu = load_inputs(batch, device_id)

    for _ in range(warmup):
        _ = run_where(cond_npu, x1_npu, x2_npu)
    torch.npu.synchronize(device_id)

    latencies_us = []
    for _ in range(repeat):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(loops):
            _ = run_where(cond_npu, x1_npu, x2_npu)
        end_event.record()
        end_event.synchronize()
        avg_us = start_event.elapsed_time(end_event) * 1000.0 / loops
        latencies_us.append(avg_us)

    stats = compute_statistics(latencies_us)

    elem_bytes = total_elems(batch) * 2
    cond_bytes = total_elems(batch) * 1
    output_bytes = elem_bytes
    total_bytes = cond_bytes + elem_bytes * 2 + output_bytes
    bw = compute_effective_bandwidth(total_bytes, 0, stats["median_us"])
    stats["effective_bandwidth_gbps"] = round(bw, 2)

    return {
        "operator": "where",
        "variant": "torch",
        "batch": batch,
        "shape": [batch] + SHAPE_TAIL,
        "dtype": "float16",
        "config": {
            "warmup": warmup,
            "loops": loops,
            "repeat": repeat,
        },
        "latency_us": stats,
        "bytes_per_elem": 5,
        "total_bytes": total_elems(batch) * 5,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline benchmark for Where")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64",
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

    output = {"results": results, "framework": "torch", "operator": "where"}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
