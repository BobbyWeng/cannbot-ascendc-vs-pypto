#!/usr/bin/env python3
"""Unified profiler runner for Cannbot: Torch, Ascend C, and PyPTO."""
import os, sys, json, subprocess, time, re

PROJECT_ROOT = "/mnt/workspace/cannbot_ascendc_vs_pypto"
BATCHES = [1, 2, 4, 8, 16, 32, 64]
WARMUP = 200
LOOPS = 100
REPEAT = 5
SHAPE_TAIL = [256, 384]


def get_msprof_cmd(output_dir):
    return [
        "/home/developer/Ascend/cann-9.0.0/bin/msprof",
        "--output=" + output_dir,
        "--ascendcl=on",
        "--ai-core=on",
        "--task-time=l0",
        "--system-call-npu=off",
        "--l2-cache=off",
    ]


def run_torch_profiler(op, batch):
    """Profile a Torch operator for given batch size."""
    impl = "torch"
    raw_dir = f"{PROJECT_ROOT}/operators/{op}/reports/raw/{impl}/b{batch}"
    os.makedirs(raw_dir, exist_ok=True)

    script = f"""
import torch, torch_npu, numpy as np, os, json, time
torch.npu.set_device(0)
DATA_DIR = "{PROJECT_ROOT}/operators/{op}/data"
SHAPE = [{batch}, 256, 384]
warmup = {WARMUP}
loops = {LOOPS}
repeat = {REPEAT}

{op}_setup = {{}}
"""

    if op == "equal":
        script += f"""
x1 = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x1_b{batch}_fp16.bin", dtype=np.float16).reshape(SHAPE)).npu(0)
x2 = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x2_b{batch}_fp16.bin", dtype=np.float16).reshape(SHAPE)).npu(0)
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
result = {{"operator": "equal", "implementation": "torch", "batch": {batch}, "loops": loops, "repeat": repeat, "raw_repeat_latency_us": latencies, "kernel_info": {{"kernel_names": ["torch_npu::eq"], "kernel_count": 1}}}}
with open("{raw_dir}/result.json", "w") as f: json.dump(result, f, indent=2)
print(f"RESULT Torch equal B={batch}: median={{sorted(latencies)[len(latencies)//2]:.1f}} us")
"""
    elif op == "where":
        script += f"""
cond = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/condition_b{batch}_bool.bin", dtype=np.uint8).reshape(SHAPE)).npu(0)
x1 = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x1_b{batch}_fp16.bin", dtype=np.float16).reshape(SHAPE)).npu(0)
x2 = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x2_b{batch}_fp16.bin", dtype=np.float16).reshape(SHAPE)).npu(0)
for _ in range(warmup): _ = torch.where(cond.bool(), x1, x2)
torch.npu.synchronize()
latencies = []
for r in range(repeat):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(loops): _ = torch.where(cond.bool(), x1, x2)
    end.record()
    torch.npu.synchronize()
    elapsed_ms = start.elapsed_time(end)
    latencies.append(elapsed_ms * 1000.0 / loops)
result = {{"operator": "where", "implementation": "torch", "batch": {batch}, "loops": loops, "repeat": repeat, "raw_repeat_latency_us": latencies, "kernel_info": {{"kernel_names": ["torch_npu::where"], "kernel_count": 1}}}}
with open("{raw_dir}/result.json", "w") as f: json.dump(result, f, indent=2)
print(f"RESULT Torch where B={batch}: median={{sorted(latencies)[len(latencies)//2]:.1f}} us")
"""
    elif op == "not":
        script += f"""
DATA_DIR = "{PROJECT_ROOT}/operators/{op}/data"
x = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x_b{batch}_all_true.bin", dtype=np.uint8).reshape(SHAPE)).npu(0)
for _ in range(warmup): _ = torch.logical_not(x)
torch.npu.synchronize()
latencies = []
for r in range(repeat):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(loops): _ = torch.logical_not(x)
    end.record()
    torch.npu.synchronize()
    elapsed_ms = start.elapsed_time(end)
    latencies.append(elapsed_ms * 1000.0 / loops)
result = {{"operator": "not", "implementation": "torch", "batch": {batch}, "loops": loops, "repeat": repeat, "raw_repeat_latency_us": latencies, "kernel_info": {{"kernel_names": ["torch_npu::logical_not"], "kernel_count": 1}}}}
with open("{raw_dir}/result.json", "w") as f: json.dump(result, f, indent=2)
print(f"RESULT Torch not B={batch}: median={{sorted(latencies)[len(latencies)//2]:.1f}} us")
"""
    elif op == "or":
        script += f"""
DATA_DIR = "{PROJECT_ROOT}/operators/{op}/data"
x1 = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x1_b{batch}_random_mask.bin", dtype=np.uint8).reshape(SHAPE)).npu(0)
x2 = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x2_b{batch}_random_mask.bin", dtype=np.uint8).reshape(SHAPE)).npu(0)
for _ in range(warmup): _ = torch.logical_or(x1, x2)
torch.npu.synchronize()
latencies = []
for r in range(repeat):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(loops): _ = torch.logical_or(x1, x2)
    end.record()
    torch.npu.synchronize()
    elapsed_ms = start.elapsed_time(end)
    latencies.append(elapsed_ms * 1000.0 / loops)
result = {{"operator": "or", "implementation": "torch", "batch": {batch}, "loops": loops, "repeat": repeat, "raw_repeat_latency_us": latencies, "kernel_info": {{"kernel_names": ["torch_npu::logical_or"], "kernel_count": 1}}}}
with open("{raw_dir}/result.json", "w") as f: json.dump(result, f, indent=2)
print(f"RESULT Torch or B={batch}: median={{sorted(latencies)[len(latencies)//2]:.1f}} us")
"""

    # Write and run
    script_path = f"{raw_dir}/profile.py"
    with open(script_path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    result = subprocess.run(
        ["python3", script_path],
        capture_output=True, text=True, timeout=600, env=env
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[:500]}")
    return result.returncode


def run_ascendc_profiler(op, batch):
    """Profile an Ascend C kernel using the built-in host binary."""
    impl = "ascendc"
    raw_dir = f"{PROJECT_ROOT}/operators/{op}/reports/raw/{impl}/b{batch}"
    os.makedirs(raw_dir, exist_ok=True)

    output_dir = f"{PROJECT_ROOT}/operators/{op}/ascendc/build/output"
    os.makedirs(output_dir, exist_ok=True)

    binary = f"{PROJECT_ROOT}/operators/{op}/ascendc/build/{op}_ascendc"
    data_dir = f"{PROJECT_ROOT}/operators/{op}/data"

    env = os.environ.copy()
    result = subprocess.run(
        [binary, "0", str(batch), "20", "8192", str(WARMUP), str(LOOPS), str(REPEAT), data_dir, output_dir],
        capture_output=True, text=True, timeout=600, env=env
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[:500]}")
        return result.returncode

    # Parse the built-in benchmark output to create result.json
    output = result.stdout
    # Extract RESULT line
    result_json = os.path.join(raw_dir, "result.json")
    stats = extract_bench_result(output, op, impl, batch)
    if stats:
        with open(result_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved -> {result_json}")

    # Save raw output
    with open(os.path.join(raw_dir, "stdout.txt"), "w") as f:
        f.write(output)

    return result.returncode


def extract_bench_result(output, op, impl, batch):
    """Parse the built-in host benchmark output."""
    stats = {
        "operator": op, "implementation": impl, "batch": batch,
        "warmup": WARMUP, "loops": LOOPS, "repeat": REPEAT
    }
    for line in output.split("\n"):
        if line.startswith("RESULT"):
            parts = line.split()
            for p in parts[1:]:
                kv = p.split("=")
                if len(kv) == 2:
                    k, v = kv
                    if k == "batch" or k == "blockDim":
                        continue
                    if k == "bw":
                        stats["bandwidth_gbs"] = float(v)
                    elif k in ("median_us", "mean_us", "min_us", "p90_us", "std_us", "cv"):
                        stats[k] = float(v)
        if line.startswith("RAW_LATENCIES:"):
            raw = line[len("RAW_LATENCIES:"):]
            stats["raw_repeat_latency_us"] = [float(x) for x in raw.split(",")]
        if line.startswith("KBYTES_RW"):
            stats["kbytes_rw_read"] = float(output.split("read=")[1].split()[0]) if "read=" in output else 0
    return stats


def run_pypto_profiler(op, batch):
    """Profile a PyPTO operator."""
    impl = "pypto"
    raw_dir = f"{PROJECT_ROOT}/operators/{op}/reports/raw/{impl}/b{batch}"
    os.makedirs(raw_dir, exist_ok=True)

    # For Not and Or, write a profiler script
    if op == "not":
        profiler_script = f"""
import os, sys, json, time
sys.path.insert(0, "{PROJECT_ROOT}/operators/not/pypto/src")
sys.path.insert(0, "{PROJECT_ROOT}/operators/not/pypto/golden")
import torch, torch_npu, numpy as np
from not_impl import not_wrapper

torch.npu.set_device(0)
DATA_DIR = "{PROJECT_ROOT}/operators/not/data"
SHAPE = [{batch}, 256, 384]
warmup = {WARMUP}
loops = {LOOPS}
repeat = {REPEAT}

x = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x_b{batch}_all_true.bin", dtype=np.uint8).reshape(SHAPE)).npu(0)

# Warmup (JIT happens here)
for _ in range(warmup): _ = not_wrapper(x)
torch.npu.synchronize()

# Timed loops
latencies = []
for r in range(repeat):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(loops): _ = not_wrapper(x)
    end.record()
    torch.npu.synchronize()
    elapsed_ms = start.elapsed_time(end)
    latencies.append(elapsed_ms * 1000.0 / loops)

result = {{
    "operator": "not", "implementation": "pypto", "batch": {batch},
    "loops": loops, "repeat": repeat,
    "raw_repeat_latency_us": latencies,
    "kernel_info": {{"kernel_names": ["pypto::logical_not"], "kernel_count": 1}}
}}
with open("{raw_dir}/result.json", "w") as f: json.dump(result, f, indent=2)
print(f"RESULT PyPTO not B={batch}: median={{sorted(latencies)[len(latencies)//2]:.1f}} us")
"""
    elif op == "or":
        profiler_script = f"""
import os, sys, json, time
sys.path.insert(0, "{PROJECT_ROOT}/operators/or/pypto/src")
sys.path.insert(0, "{PROJECT_ROOT}/operators/or/pypto/golden")
import torch, torch_npu, numpy as np
from or_impl import or_wrapper

torch.npu.set_device(0)
DATA_DIR = "{PROJECT_ROOT}/operators/or/data"
SHAPE = [{batch}, 256, 384]
warmup = {WARMUP}
loops = {LOOPS}
repeat = {REPEAT}

x1 = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x1_b{batch}_random_mask.bin", dtype=np.uint8).reshape(SHAPE)).npu(0)
x2 = torch.from_numpy(np.fromfile(f"{{DATA_DIR}}/x2_b{batch}_random_mask.bin", dtype=np.uint8).reshape(SHAPE)).npu(0)

for _ in range(warmup): _ = or_wrapper(x1, x2)
torch.npu.synchronize()

latencies = []
for r in range(repeat):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(loops): _ = or_wrapper(x1, x2)
    end.record()
    torch.npu.synchronize()
    elapsed_ms = start.elapsed_time(end)
    latencies.append(elapsed_ms * 1000.0 / loops)

result = {{
    "operator": "or", "implementation": "pypto", "batch": {batch},
    "loops": loops, "repeat": repeat,
    "raw_repeat_latency_us": latencies,
    "kernel_info": {{"kernel_names": ["pypto::bitwise_or"], "kernel_count": 1}}
}}
with open("{raw_dir}/result.json", "w") as f: json.dump(result, f, indent=2)
print(f"RESULT PyPTO or B={batch}: median={{sorted(latencies)[len(latencies)//2]:.1f}} us")
"""
    else:
        print(f"PyPTO profiler not supported for {op} (BLOCKED_BACKEND)")
        return 0

    script_path = f"{raw_dir}/profile.py"
    with open(script_path, "w") as f:
        f.write(profiler_script)
    env = os.environ.copy()
    result = subprocess.run(
        ["python3", script_path],
        capture_output=True, text=True, timeout=600, env=env
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[:500]}")
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run_profiler.py <operator> <implementation> <batch>")
        sys.exit(1)

    op = sys.argv[1]
    impl = sys.argv[2]
    batch = int(sys.argv[3])

    if impl == "torch":
        sys.exit(run_torch_profiler(op, batch))
    elif impl == "ascendc":
        sys.exit(run_ascendc_profiler(op, batch))
    elif impl == "pypto":
        sys.exit(run_pypto_profiler(op, batch))
    else:
        print(f"Unknown implementation: {impl}")
        sys.exit(1)
