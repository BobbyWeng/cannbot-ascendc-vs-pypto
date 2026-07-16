
import torch, torch_npu, numpy as np, os, json, time
torch.npu.set_device(0)
DATA_DIR = "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/equal/data"
SHAPE = [2, 256, 384]
warmup = 200
loops = 100
repeat = 5

equal_setup = {}

x1 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x1_b2_fp16.bin", dtype=np.float16).reshape(SHAPE)).npu(0)
x2 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x2_b2_fp16.bin", dtype=np.float16).reshape(SHAPE)).npu(0)
for _ in range(warmup): _ = torch.eq(x1, x2)
torch.npu.synchronize()
latencies = []
for r in range(repeat):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(loops): _ = torch.eq(x1, x2)
    end.record()
    torch.npu.synchronize()
    elapsed_ms = start.elapsed_time(end)
    latencies.append(elapsed_ms * 1000.0 / loops)
result = {"operator": "equal", "implementation": "torch", "batch": 2, "loops": loops, "repeat": repeat, "raw_repeat_latency_us": latencies, "kernel_info": {"kernel_names": ["torch_npu::eq"], "kernel_count": 1}}
with open("/mnt/workspace/cannbot_ascendc_vs_pypto/operators/equal/reports/raw/torch/b2/result.json", "w") as f: json.dump(result, f, indent=2)
print(f"RESULT Torch equal B=2: median={sorted(latencies)[len(latencies)//2]:.1f} us")
