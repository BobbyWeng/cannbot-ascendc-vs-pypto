#!/usr/bin/env python3
"""MatMul correctness checker. Compares output BINs against reference.

Usage:
  python3 data/generation_scripts/correctness.py ascendc B=1 fp16
  python3 data/generation_scripts/correctness.py ascendc B=1 fp32
  python3 data/generation_scripts/correctness.py torch
"""
import os, sys, json, argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.correctness import check_correctness

BATCHES = [1, 2, 4, 8, 16, 32]
HEADS = 12
M, N, K = 256, 32, 256
SPECIAL_CASES = [
    "perf_fp16", "zeros", "ones", "identity", "sparse", "small_values",
    "mixed_signs", "cancellation", "max_safe", "underflow",
    "overflow_risk", "nan", "inf",
]


def check_batch(impl, batch, ref_type="fp16"):
    """Check correctness for a single batch."""
    results = {}
    all_pass = True

    for case in SPECIAL_CASES:
        ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_{case}_{ref_type}.bin")

        if impl == "ascendc":
            out_path = os.path.join(PROJECT_ROOT, f"operators/matmul/ascendc/build/output/output_b{batch}_{case}.bin")
        elif impl == "torch":
            out_path = os.path.join(DATA_DIR, f"torch_output_b{batch}_{case}.bin")
        elif impl == "pypto":
            out_path = os.path.join(DATA_DIR, f"pypto_output_b{batch}_{case}.bin")
        else:
            raise ValueError(f"Unknown impl: {impl}")

        if not os.path.exists(out_path):
            results[case] = {"status": "SKIP", "error": f"Output not found: {out_path}"}
            all_pass = False
            continue
        if not os.path.exists(ref_path):
            results[case] = {"status": "SKIP", "error": f"Reference not found: {ref_path}"}
            all_pass = False
            continue

        shape = [batch, HEADS, M, N]
        output = torch.from_numpy(np.fromfile(out_path, dtype=np.float16).reshape(shape))
        reference = torch.from_numpy(np.fromfile(ref_path, dtype=np.float16).reshape(shape))

        result = check_correctness(
            output, reference,
            rtol=0.01, atol=0.01,
            require_bitwise=False,
            label=f"{impl} B={batch} case={case}"
        )
        results[case] = result
        if result["status"] == "FAIL":
            all_pass = False
            print(f"  FAIL B={batch} case={case}: max_abs={result['max_abs_diff']:.6f} max_rel={result['max_rel_diff']:.6f}")
        else:
            print(f"  PASS B={batch} case={case}")

    return {"batch": batch, "impl": impl, "results": results, "all_pass": all_pass}


def main():
    parser = argparse.ArgumentParser(description="MatMul correctness checker")
    parser.add_argument("impl", choices=["ascendc", "torch", "pypto"])
    parser.add_argument("--batches", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--ref-type", type=str, default="fp16", choices=["fp16", "fp32"])
    args = parser.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    all_pass = True
    batch_results = []

    for b in batches:
        result = check_batch(args.impl, b, args.ref_type)
        batch_results.append(result)
        if not result["all_pass"]:
            all_pass = False

    output = {
        "operator": "matmul",
        "implementation": args.impl,
        "reference_type": args.ref_type,
        "batch_results": batch_results,
        "all_pass": all_pass,
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "..", "reports", "correctness",
                            f"correctness_{args.impl}_{args.ref_type}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"Results saved to {out_path}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
