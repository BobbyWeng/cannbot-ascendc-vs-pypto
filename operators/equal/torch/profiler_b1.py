#!/usr/bin/env python3
import torch
import torch_npu
import numpy as np
import os
import json

torch.npu.set_device(0)
DATA_DIR = "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/equal/data"
output_dir = "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/equal/reports/raw/torch/b1"
os.makedirs(output_dir, exist_ok=True)

B = 1
shape = [B, 256, 384]
x1 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x1_b{B}_fp16.bin", dtype=np.float16).reshape(shape))
x2 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x2_b{B}_fp16.bin", dtype=np.float16).reshape(shape))
x1_npu = x1.npu(0)
x2_npu = x2.npu(0)

for _ in range(200):
    _ = torch.eq(x1_npu, x2_npu)
torch.npu.synchronize()

start = torch.npu.Event(enable_timing=True)
end = torch.npu.Event(enable_timing=True)
latencies = []
for _ in range(100):
    start.record()
    _ = torch.eq(x1_npu, x2_npu)
    end.record()
    torch.npu.synchronize()
    latencies.append(start.elapsed_time(end) * 1000.0)

latencies.sort()
result = {
    "operator": "equal",
    "implementation": "torch",
    "batch": B,
    "kernel_info": {"kernel_count": 1, "kernel_names": ["torch_npu::eq"]},
    "latency_stats": {
        "median_us": latencies[len(latencies)//2],
        "mean_us": sum(latencies)/len(latencies),
        "min_us": latencies[0],
        "p90_us": latencies[int(len(latencies)*0.9)],
        "std_us": (sum((x - sum(latencies)/len(latencies))**2 for x in latencies)/len(latencies))**0.5
    }
}
result["latency_stats"]["cv"] = result["latency_stats"]["std_us"] / result["latency_stats"]["mean_us"] * 100.0

with open(f"{output_dir}/result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Equal Torch B=1: median={result['latency_stats']['median_us']:.1f} us")
