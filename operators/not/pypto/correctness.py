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
from not_impl import not_wrapper

SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

BOUNDARY_CASES = ["all_true", "all_false", "alternating", "random_mask", "sparse_true", "dense_true"]


def run_correctness(batch, case_name, device_id):
    torch.npu.set_device(device_id)
    shape = [batch] + SHAPE_TAIL
    x = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"x_b{batch}_{case_name}.bin"), dtype=np.uint8).reshape(shape))
    ref = torch.from_numpy(np.fromfile(os.path.join(DATA_DIR, f"reference_b{batch}_{case_name}.bin"), dtype=np.uint8).reshape(shape))

    y = not_wrapper(x.npu(device_id))
    torch.npu.synchronize(device_id)

    ref_npu = ref.npu(device_id)
    result = check_correctness(y, ref_npu, rtol=0, atol=0, require_bitwise=True, label=f"B={batch} {case_name}")
    result["batch"] = batch
    result["case"] = case_name
    result["shape"] = shape
    result["dtype"] = "bool (uint8)"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyPTO correctness for Not")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    all_pass = True
    results = []
    for b in batches:
        for case in BOUNDARY_CASES:
            result = run_correctness(b, case, args.device)
            results.append(result)
            status = result["status"]
            if status == "PASS":
                print(f"  B={b} {case}: PASS")
            else:
                print(f"  B={b} {case}: FAIL (mismatches={result.get('bitwise_mismatch_count', '?')})")
                all_pass = False

    output = {"operator": "not", "variant": "pypto", "results": results, "all_pass": all_pass}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.exit(0 if all_pass else 1)
