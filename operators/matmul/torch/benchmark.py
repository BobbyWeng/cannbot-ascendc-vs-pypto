#!/usr/bin/env python3
"""Torch baseline benchmark for MatMul [B,12,256,256] @ [B,12,256,32]."""
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
import torch
import torch_npu
from common.benchmark import compute_statistics, compute_effective_bandwidth

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HEADS = 12
M, N, K = 256, 32, 256


def total_matrices(batch):
    return batch * HEADS


def flops_per_op(batch):
    return 2 * batch * HEADS * M * N * K


def input_bytes(batch):
    return batch * HEADS * (M * K + K * N) * 2  # FP16


def output_bytes(batch):
    return batch * HEADS * M * N * 2


def run_benchmark(batch, warmup, loops, repeat, device_id):
    torch.npu.set_device(device_id)

    a_path = os.path.join(DATA_DIR, f"A_b{batch}_perf_fp16.bin")
    b_path = os.path.join(DATA_DIR, f"B_b{batch}_perf_fp16.bin")

    shape_a = [batch, HEADS, M, K]
    shape_b = [batch, HEADS, K, N]
    A = torch.from_numpy(np.fromfile(a_path, dtype=np.float16).reshape(shape_a)).npu(device_id)
    B_mat = torch.from_numpy(np.fromfile(b_path, dtype=np.float16).reshape(shape_b)).npu(device_id)

    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(A.half(), B_mat.half())
    torch.npu.synchronize(device_id)

    # Timed loops
    latencies_us = []
    for _ in range(repeat):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(loops):
            _ = torch.matmul(A.half(), B_mat.half())
        end_event.record()
        end_event.synchronize()
        avg_us = start_event.elapsed_time(end_event) * 1000.0 / loops
        latencies_us.append(avg_us)

    stats = compute_statistics(latencies_us)
    total_flops = flops_per_op(batch)
    tflops = total_flops / stats["median_us"] / 1e6  # TFLOPS

    return {
        "operator": "matmul",
        "variant": "torch",
        "batch": batch,
        "shape": {"A": [batch, HEADS, M, K], "B": [batch, HEADS, K, N], "Y": [batch, HEADS, M, N]},
        "dtype": "float16",
        "config": {"warmup": warmup, "loops": loops, "repeat": repeat},
        "latency_us": stats,
        "flops_per_op": total_flops,
        "achieved_tflops": round(tflops, 4),
        "input_bytes": input_bytes(batch),
        "output_bytes": output_bytes(batch),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline benchmark for MatMul")
    parser.add_argument("--batches", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    results = []
    for b in batches:
        result = run_benchmark(b, args.warmup, args.loops, args.repeat, args.device)
        results.append(result)
        print(f"  B={b}: median={result['latency_us']['median_us']:.1f} us, TFLOPS={result['achieved_tflops']:.4f}")

    output = {"results": results, "framework": "torch", "operator": "matmul"}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
