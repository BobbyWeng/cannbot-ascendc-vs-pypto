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
from common.correctness import check_correctness
from add_impl import add_4

SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def run_correctness(batch, device_id):
    torch.npu.set_device(device_id)
    shape = [batch] + SHAPE_TAIL
    x1 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x1_b{batch}_fp16.bin"), dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x2_b{batch}_fp16.bin"), dtype=np.float16).reshape(shape))
    x3 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x3_b{batch}_fp16.bin"), dtype=np.float16).reshape(shape))
    x4 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x4_b{batch}_fp16.bin"), dtype=np.float16).reshape(shape))
    ref = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"reference_b{batch}_fp16.bin"), dtype=np.float16).reshape(shape))

    y = add_4(x1.npu(device_id), x2.npu(device_id), x3.npu(device_id), x4.npu(device_id))
    torch.npu.synchronize(device_id)

    ref_npu = ref.npu(device_id)
    result = check_correctness(y, ref_npu, label=f"B={batch}")
    result["batch"] = batch
    result["shape"] = shape
    result["dtype"] = "float16"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyPTO correctness for 4-input Add")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    all_pass = True
    results = []
    for b in batches:
        result = run_correctness(b, args.device)
        results.append(result)
        status = result["status"]
        if status == "PASS":
            print(f"  B={b}: PASS")
        else:
            print(f"  B={b}: FAIL (mismatches={result.get('bitwise_mismatch_count', '?')})")
            all_pass = False

    output = {"operator": "add", "variant": "pypto", "results": results, "all_pass": all_pass}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.exit(0 if all_pass else 1)
