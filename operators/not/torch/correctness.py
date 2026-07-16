#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
import torch
import torch_npu
from common.correctness import check_correctness

SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

BOUNDARY_CASES = ["all_true", "all_false", "alternating", "random_mask", "sparse_true", "dense_true"]


def run_correctness(batch, case_name, device_id):
    torch.npu.set_device(device_id)
    x_path = os.path.join(DATA_DIR, f"x_b{batch}_{case_name}.bin")
    ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_{case_name}.bin")
    shape = [batch] + SHAPE_TAIL

    if not os.path.exists(x_path) or not os.path.exists(ref_path):
        return {"batch": batch, "case": case_name, "status": "SKIP", "error": "Missing data files"}

    x = torch.from_numpy(np.fromfile(x_path, dtype=np.uint8).reshape(shape))
    ref = torch.from_numpy(np.fromfile(ref_path, dtype=np.uint8).reshape(shape))

    x_npu = x.npu(device_id)
    ref_npu = ref.npu(device_id)

    output = torch.logical_not(x_npu.bool()).byte()
    torch.npu.synchronize(device_id)

    result = check_correctness(output, ref_npu, rtol=0, atol=0, require_bitwise=True, label=f"B={batch} {case_name}")
    result["batch"] = batch
    result["case"] = case_name
    result["shape"] = [batch] + SHAPE_TAIL
    result["dtype"] = "bool (uint8)"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline correctness for Not")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
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
            elif status == "SKIP":
                print(f"  B={b} {case}: SKIP ({result.get('error', '')})")
            else:
                print(f"  B={b} {case}: FAIL (mismatches={result.get('bitwise_mismatch_count', '?')})")
                all_pass = False

    output = {
        "operator": "not",
        "variant": "torch",
        "results": results,
        "all_pass": all_pass,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.exit(0 if all_pass else 1)
