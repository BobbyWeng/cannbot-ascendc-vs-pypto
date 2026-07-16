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
OUTPUT_TAIL = [256]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

COVERAGE_CASES = [
    "random_finite", "all_zero", "all_one", "pos_neg_cancel",
    "small_values", "large_values", "overflow_risk", "underflow_risk",
    "nan", "inf",
]


def run_correctness(batch, case_name, device_id):
    torch.npu.set_device(device_id)
    x_path = os.path.join(DATA_DIR, f"x_b{batch}_{case_name}.bin")
    ref_fp32_path = os.path.join(DATA_DIR, f"reference_b{batch}_{case_name}_fp32_accum.bin")
    ref_fp16_path = os.path.join(DATA_DIR, f"reference_b{batch}_{case_name}_fp16_path.bin")
    shape = [batch] + SHAPE_TAIL

    if not os.path.exists(x_path):
        return {"batch": batch, "case": case_name, "status": "SKIP", "error": "Missing data files"}

    x = torch.from_numpy(np.fromfile(x_path, dtype=np.float16).reshape(shape))
    ref_fp32 = torch.from_numpy(np.fromfile(ref_fp32_path, dtype=np.float16).reshape([batch] + OUTPUT_TAIL))
    ref_fp16 = torch.from_numpy(np.fromfile(ref_fp16_path, dtype=np.float16).reshape([batch] + OUTPUT_TAIL))

    x_npu = x.npu(device_id)
    ref_fp32_npu = ref_fp32.npu(device_id)
    ref_fp16_npu = ref_fp16.npu(device_id)

    output = torch.sum(x_npu, dim=-1)
    torch.npu.synchronize(device_id)

    result_fp32 = check_correctness(output, ref_fp32_npu, rtol=0.01, atol=0.01, require_bitwise=False, label=f"B={batch} {case_name} fp32-accum")
    result_fp16 = check_correctness(output, ref_fp16_npu, rtol=0.01, atol=0.01, require_bitwise=False, label=f"B={batch} {case_name} fp16-path")

    result = {
        "batch": batch,
        "case": case_name,
        "shape": [batch] + SHAPE_TAIL,
        "dtype": "float16",
        "output_actual_dtype": str(output.dtype),
        "fp32_accum": result_fp32,
        "fp16_path": result_fp16,
    }

    if result_fp32["status"] == "PASS" or result_fp16["status"] == "PASS":
        if result_fp32["status"] == "PASS":
            result["status"] = "PASS"
            result["match_reference"] = "fp32_accum"
        else:
            result["status"] = "PASS"
            result["match_reference"] = "fp16_path"
    else:
        result["status"] = "FAIL"

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline correctness for ReduceSum")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    all_pass = True
    results = []
    for b in batches:
        for case in COVERAGE_CASES:
            result = run_correctness(b, case, args.device)
            results.append(result)
            status = result["status"]
            if status == "PASS":
                print(f"  B={b} {case}: PASS (match={result.get('match_reference', 'N/A')})")
            elif status == "SKIP":
                print(f"  B={b} {case}: SKIP ({result.get('error', '')})")
            else:
                print(f"  B={b} {case}: FAIL")
                all_pass = False

    output = {
        "operator": "reduce_sum",
        "variant": "torch",
        "results": results,
        "all_pass": all_pass,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.exit(0 if all_pass else 1)
