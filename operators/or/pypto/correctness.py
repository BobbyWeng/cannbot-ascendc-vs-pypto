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
from or_impl import or_wrapper

SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

INPUT_COMBINATIONS = [
    "false_false", "false_true", "true_false", "true_true",
    "random_mask", "sparse", "dense",
]


def run_correctness(batch, variant, device_id):
    torch.npu.set_device(device_id)
    shape = [batch] + SHAPE_TAIL
    x1 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x1_b{batch}_{variant}.bin"), dtype=np.uint8).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x2_b{batch}_{variant}.bin"), dtype=np.uint8).reshape(shape))
    ref = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"reference_b{batch}_{variant}.bin"), dtype=np.uint8).reshape(shape))

    y = or_wrapper(x1.npu(device_id), x2.npu(device_id))
    torch.npu.synchronize(device_id)

    ref_npu = ref.npu(device_id)
    result = check_correctness(y, ref_npu, label=f"B={batch} {variant}")
    result["batch"] = batch
    result["variant"] = variant
    result["shape"] = shape
    result["dtype"] = "bool"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyPTO correctness for OR")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--variant", type=str, default=",".join(INPUT_COMBINATIONS))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    variants = [v.strip() for v in args.variant.split(",")]
    all_pass = True
    results = []
    for b in batches:
        for v in variants:
            result = run_correctness(b, v, args.device)
            results.append(result)
            status = result["status"]
            if status == "PASS":
                print(f"  B={b} {v}: PASS")
            else:
                print(f"  B={b} {v}: FAIL (mismatches={result.get('bitwise_mismatch_count', '?')})")
                all_pass = False

    output = {"operator": "or", "variant": "pypto", "results": results, "all_pass": all_pass}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.exit(0 if all_pass else 1)
