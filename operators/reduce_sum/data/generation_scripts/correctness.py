import numpy as np
import os
import sys

SHAPE_TAIL = [256, 384]
OUTPUT_TAIL = [256]

BATCHES = [1, 2, 4, 8, 16, 32, 64]
COVERAGE_CASES = [
    "random_finite",
    "all_zero",
    "all_one",
    "pos_neg_cancel",
    "small_values",
    "large_values",
    "overflow_risk",
    "underflow_risk",
    "nan",
    "inf",
]


def check_correctness(output_path, reference_path, reference_desc, batch, verbose=True):
    output = np.fromfile(output_path, dtype=np.float16).reshape([batch] + OUTPUT_TAIL)
    reference = np.fromfile(reference_path, dtype=np.float16).reshape([batch] + OUTPUT_TAIL)

    if output.shape != reference.shape:
        return {
            "status": "FAIL",
            "error": f"Shape mismatch: output={output.shape}, reference={reference.shape}"
        }

    abs_diff = np.abs(output.astype(np.float32) - reference.astype(np.float32))
    rel_diff = abs_diff / (np.abs(reference.astype(np.float32)) + 1e-12)
    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())

    mismatches = int(np.sum(abs_diff > 0.01))
    total = int(output.size)

    result = {
        "status": "PASS" if max_abs <= 0.01 else "FAIL",
        "reference": reference_desc,
        "total_elements": total,
        "mismatch_count": mismatches,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
    }

    if verbose:
        print(f"  [{reference_desc}] status={result['status']} max_abs={max_abs:.6f} max_rel={max_rel:.6f} mismatches={mismatches}/{total}")

    return result


def check_all(output_dir, data_dir):
    results = {}
    all_pass = True
    for b in BATCHES:
        for case in COVERAGE_CASES:
            out_path = os.path.join(output_dir, f"output_b{b}_{case}.bin")
            ref_fp32_path = os.path.join(data_dir, f"reference_b{b}_{case}_fp32_accum.bin")
            ref_fp16_path = os.path.join(data_dir, f"reference_b{b}_{case}_fp16_path.bin")

            key = f"b{b}_{case}"
            if not os.path.exists(out_path):
                print(f"  [{key}] output not found at {out_path}")
                results[key] = {"status": "SKIP", "error": "output file not found"}
                all_pass = False
                continue

            r_fp32 = check_correctness(out_path, ref_fp32_path, "fp32_accum", b, verbose=False)
            r_fp16 = check_correctness(out_path, ref_fp16_path, "fp16_path", b, verbose=False)

            results[key] = {
                "fp32_accum": r_fp32,
                "fp16_path": r_fp16,
                "overall": "PASS" if r_fp32["status"] == "PASS" else "FAIL"
            }

            if r_fp32["status"] == "PASS":
                print(f"  [{key}]: PASS (vs fp32_accum)")
            else:
                print(f"  [{key}]: FAILED (vs fp32_accum: max_abs={r_fp32['max_abs_diff']:.6f})")
                all_pass = False

    return all_pass, results


if __name__ == "__main__":
    op_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(op_dir, "ascendc", "build", "output")
    data_dir = os.path.join(op_dir, "data")
    print("ReduceSum operator correctness check:")
    all_pass, results = check_all(output_dir, data_dir)
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)
