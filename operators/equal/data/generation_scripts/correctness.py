import numpy as np
import hashlib
import os
import json

SHAPE_TAIL = [256, 384]


def check_correctness_bool(output_path, reference_path, batch, verbose=True):
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

    mismatch_mask = output != reference
    mismatch_count = int(np.sum(mismatch_mask))

    equal = (mismatch_count == 0)

    result = {
        "status": "PASS" if equal else "FAIL",
        "total_elements": int(total_elements),
        "checked_elements": int(output.size),
        "bitwise_equal": equal,
        "bitwise_mismatch_count": mismatch_count,
        "true_in_output": int(np.sum(output == 1)),
        "false_in_output": int(np.sum(output == 0)),
        "true_in_reference": int(np.sum(reference == 1)),
        "false_in_reference": int(np.sum(reference == 0)),
    }

    if verbose:
        print(f"  Correctness: {result['status']}")
        print(f"    Elements checked: {result['checked_elements']}")
        print(f"    Bitwise equal: {result['bitwise_equal']}")
        print(f"    Mismatches: {result['bitwise_mismatch_count']}")
        print(f"    True in output: {result['true_in_output']}")
        print(f"    False in output: {result['false_in_output']}")

    return result


def check_all_batches(output_dir, data_dir, batches, variant_name):
    results = {}
    all_pass = True
    for b in batches:
        out_path = os.path.join(output_dir, f"output_b{b}.bin")
        ref_path = os.path.join(data_dir, f"reference_b{b}_bool.bin")
        if not os.path.exists(out_path):
            print(f"  [{variant_name}] b={b}: output not found at {out_path}")
            results[b] = {"status": "SKIP", "error": "output file not found"}
            all_pass = False
            continue
        result = check_correctness_bool(out_path, ref_path, b, verbose=False)
        results[b] = result
        if result["status"] != "PASS":
            all_pass = False
            print(f"  [{variant_name}] b={b}: FAILED ({result['bitwise_mismatch_count']} mismatches)")
        else:
            print(f"  [{variant_name}] b={b}: PASS")
    return all_pass, results


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: correctness.py <output_path> <reference_path> <batch>")
        sys.exit(1)
    result = check_correctness_bool(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    sys.exit(0 if result["status"] == "PASS" else 1)
