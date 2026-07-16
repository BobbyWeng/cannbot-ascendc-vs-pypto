import numpy as np
import os
import sys

SHAPE_TAIL = [256, 384]
BATCHES = [1, 2, 4, 8, 16, 32, 64]
BOUNDARY_CASES = [
    "all_true",
    "all_false",
    "alternating",
    "random_mask",
    "sparse_true",
    "dense_true",
]


def check_correctness(output_path, reference_path, batch, verbose=True):
    output = np.fromfile(output_path, dtype=np.uint8)
    reference = np.fromfile(reference_path, dtype=np.uint8)

    total_elements = int(np.prod([batch] + SHAPE_TAIL))
    if output.size != reference.size:
        return {
            "status": "FAIL",
            "error": f"Size mismatch: output={output.size}, reference={reference.size}, expected={total_elements}"
        }

    if output.size != total_elements:
        return {
            "status": "FAIL",
            "error": f"Expected {total_elements} elements, got {output.size}"
        }

    mismatches = int(np.sum(output != reference))
    max_val = int(np.max(output))
    min_val = int(np.min(output))

    result = {
        "status": "PASS" if mismatches == 0 else "FAIL",
        "total_elements": int(total_elements),
        "checked_elements": int(output.size),
        "mismatch_count": mismatches,
        "output_max": max_val,
        "output_min": min_val,
    }

    if verbose:
        print(f"  Correctness: {result['status']}")
        print(f"    Elements checked: {result['checked_elements']}")
        print(f"    Mismatches: {result['mismatch_count']}")
        print(f"    Output range: [{result['output_min']}, {result['output_max']}]")

    return result


def check_all(output_dir, data_dir):
    results = {}
    all_pass = True
    for b in BATCHES:
        for case in BOUNDARY_CASES:
            out_path = os.path.join(output_dir, f"output_b{b}_{case}.bin")
            ref_path = os.path.join(data_dir, f"reference_b{b}_{case}.bin")
            key = f"b{b}_{case}"
            if not os.path.exists(out_path):
                print(f"  [{key}] output not found at {out_path}")
                results[key] = {"status": "SKIP", "error": "output file not found"}
                all_pass = False
                continue
            result = check_correctness(out_path, ref_path, b, verbose=False)
            results[key] = result
            if result["status"] != "PASS":
                all_pass = False
                print(f"  [{key}]: FAILED ({result['mismatch_count']} mismatches)")
            else:
                print(f"  [{key}]: PASS")
    return all_pass, results


if __name__ == "__main__":
    op_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(op_dir, "ascendc", "build", "output")
    data_dir = os.path.join(op_dir, "data")
    print("NOT operator correctness check:")
    all_pass, results = check_all(output_dir, data_dir)
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)
