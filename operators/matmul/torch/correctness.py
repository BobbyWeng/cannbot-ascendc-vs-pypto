#!/usr/bin/env python3
"""Torch baseline correctness for MatMul [B,12,256,256] @ [B,12,256,32]."""
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
import torch
import torch_npu
from common.correctness import check_correctness


def check_tolerance(output, reference, atol=0.03125, rtol=1.5):
    """Simple tolerance check that handles NaN/Inf and sparse cases."""
    if output.shape != reference.shape:
        return {"status": "FAIL", "error": "shape mismatch"}

    mask = ~(torch.isnan(reference) | torch.isinf(reference))
    if mask.sum() == 0:
        return {"status": "PASS", "note": "all reference values are NaN/Inf"}

    abs_diff = (output[mask].float() - reference[mask].float()).abs()
    rel_diff = abs_diff / (reference[mask].float().abs() + 1e-12)

    max_abs = float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0
    max_rel = float(rel_diff.max().item()) if rel_diff.numel() > 0 else 0.0

    # NaN check: output should have NaN at same positions as reference
    nan_mismatch = (torch.isnan(output) != torch.isnan(reference)).sum().item()
    inf_mismatch = (torch.isinf(output) != torch.isinf(reference)).sum().item()

    passed = (max_abs <= atol or max_rel <= rtol) and nan_mismatch == 0 and inf_mismatch == 0

    return {
        "status": "PASS" if passed else "FAIL",
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "nan_mismatch": nan_mismatch,
        "inf_mismatch": inf_mismatch,
        "checked_elements": int(mask.sum().item()),
        "atol": atol,
        "rtol": rtol,
    }

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BATCHES = [1, 2, 4, 8, 16, 32]
HEADS = 12
M, N, K = 256, 32, 256
SPECIAL_CASES = [
    "perf_fp16", "zeros", "ones", "identity", "sparse", "small_values",
    "mixed_signs", "cancellation", "max_safe", "underflow",
    "overflow_risk", "nan", "inf",
]


def run_correctness(batch, device_id, ref_type="fp16"):
    torch.npu.set_device(device_id)
    results = []
    all_pass = True

    for case in SPECIAL_CASES:
        a_path = os.path.join(DATA_DIR, f"A_b{batch}_{case}.bin")
        b_path = os.path.join(DATA_DIR, f"B_b{batch}_{case}.bin")
        ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_{case}_{ref_type}.bin")

        if not all(os.path.exists(p) for p in [a_path, b_path, ref_path]):
            results.append({"case": case, "batch": batch, "status": "SKIP", "error": "missing data file"})
            continue

        shape_a = [batch, HEADS, M, K]
        shape_b = [batch, HEADS, K, N]
        A = torch.from_numpy(np.fromfile(a_path, dtype=np.float16).reshape(shape_a)).npu(device_id)
        B_mat = torch.from_numpy(np.fromfile(b_path, dtype=np.float16).reshape(shape_b)).npu(device_id)
        ref = torch.from_numpy(np.fromfile(ref_path, dtype=np.float16).reshape(batch, HEADS, M, N))

        output = torch.matmul(A.half(), B_mat.half())
        torch.npu.synchronize(device_id)

        result = check_tolerance(output.cpu(), ref)
        result["case"] = case
        result["batch"] = batch
        results.append(result)

        if result["status"] == "FAIL":
            all_pass = False
            print(f"  {case}: FAIL (max_abs={result['max_abs_diff']:.6f} max_rel={result['max_rel_diff']:.6f})")
        else:
            print(f"  {case}: PASS")

    return {"batch": batch, "impl": "torch", "results": results, "all_pass": all_pass}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline correctness for MatMul")
    parser.add_argument("--batches", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ref-type", type=str, default="fp16")
    args = parser.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    batch_results = []
    all_pass = True

    # Use FP16 reference for tolerance-based comparison
    # FP32 reference comparison is diagnostic-only since FP16 accumulation differs
    for b in batches:
        result = run_correctness(b, args.device, "fp16")
        batch_results.append(result)
        if not result["all_pass"]:
            all_pass = False

    output = {
        "operator": "matmul",
        "variant": "torch",
        "reference_type": args.ref_type,
        "batch_results": batch_results,
        "all_pass": all_pass,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.exit(0 if all_pass else 1)
