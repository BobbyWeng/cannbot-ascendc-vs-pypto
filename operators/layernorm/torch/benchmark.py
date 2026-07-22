#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(PROJECT_DIR, "common"))
import torch
import torch_npu
from benchmark import compute_statistics, compute_effective_bandwidth

NORM_SHAPE = [32]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def total_elems(batch):
    return batch * 256 * 32

def load_input(batch, device_id):
    x_path = os.path.join(DATA_DIR, f"input_b{batch}_fp16.bin")
    w_path = os.path.join(DATA_DIR, "weight_fp16.bin")
    b_path = os.path.join(DATA_DIR, "bias_fp16.bin")
    shape = [batch] + NORM_SHAPE
    x = torch.from_numpy(np.fromfile(x_path, dtype=np.float16).reshape(shape))
    w = torch.from_numpy(np.fromfile(w_path, dtype=np.float16).reshape(-1))
    b = torch.from_numpy(np.fromfile(b_path, dtype=np.float16).reshape(-1))
    return x.npu(device_id), w.npu(device_id), b.npu(device_id)

def run_benchmark(batch, warmup, loops, repeat, device_id):
    torch.npu.set_device(device_id)
    x_npu, w_npu, b_npu = load_input(batch, device_id)
    eps = 1e-5

    for _ in range(warmup):
        _ = torch.nn.functional.layer_norm(x_npu, NORM_SHAPE, weight=w_npu, bias=b_npu, eps=eps)
    torch.npu.synchronize(device_id)

    latencies_us = []
    for _ in range(repeat):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(loops):
            _ = torch.nn.functional.layer_norm(x_npu, NORM_SHAPE, weight=w_npu, bias=b_npu, eps=eps)
        end_event.record()
        end_event.synchronize()
        avg_us = start_event.elapsed_time(end_event) * 1000.0 / loops
        latencies_us.append(avg_us)

    stats = compute_statistics(latencies_us)

    elem_bytes = total_elems(batch) * 2
    total_bytes = elem_bytes * 2  # read x, write y
    bw = compute_effective_bandwidth(total_bytes, total_bytes, stats["median_us"])
    stats["effective_bandwidth_gbps"] = round(bw, 2)

    return {
        "operator": "layernorm",
        "variant": "torch",
        "batch": batch,
        "shape": [batch] + NORM_SHAPE,
        "dtype": "float16",
        "config": {
            "warmup": warmup,
            "loops": loops,
            "repeat": repeat,
        },
        "latency_us": stats,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline benchmark for LayerNorm")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations")
    parser.add_argument("--loops", type=int, default=100, help="Inner loop iterations per repeat")
    parser.add_argument("--repeat", type=int, default=10, help="Number of repeat measurements")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    results = []
    for b in batches:
        result = run_benchmark(b, args.warmup, args.loops, args.repeat, args.device)
        results.append(result)
        print(json.dumps(result, indent=2))

    output = {"results": results, "framework": "torch", "operator": "layernorm"}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
