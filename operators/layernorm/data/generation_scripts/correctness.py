import numpy as np
import hashlib
import os

NORM_SHAPE = [256, 32]

EPS = 1e-5

def check_correctness(output_path, reference_path, batch, rtol=0.001, atol=0.01, verbose=True):
    output = np.fromfile(output_path, dtype=np.float16)
    reference = np.fromfile(reference_path, dtype=np.float16)

    total_elements = int(np.prod([batch] + NORM_SHAPE))
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

    nan_count = int(np.sum(np.isnan(output)))
    inf_count = int(np.sum(np.isinf(output)))

    output_f32 = output.astype(np.float32)
    ref_f32 = reference.astype(np.float32)
    abs_diff = np.abs(output_f32 - ref_f32)
    rel_diff = abs_diff / (np.abs(ref_f32) + 1e-12)

    max_abs_diff = float(np.max(abs_diff))
    max_rel_diff = float(np.max(rel_diff))

    within_tol = (abs_diff <= atol) | (rel_diff <= rtol)
    mismatch_count = int(np.sum(~within_tol))

    passed = mismatch_count == 0 and nan_count == 0 and inf_count == 0

    result = {
        "status": "PASS" if passed else "FAIL",
        "total_elements": int(total_elements),
        "checked_elements": int(output.size),
        "mismatch_count": mismatch_count,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "atol": atol,
        "rtol": rtol,
    }

    if verbose:
        print(f"  Correctness: {result['status']}")
        print(f"    Elements: {result['checked_elements']}")
        print(f"    Mismatches: {result['mismatch_count']}")
        print(f"    Max abs diff: {result['max_abs_diff']:.6e}")
        print(f"    Max rel diff: {result['max_rel_diff']:.6e}")
        print(f"    NaN count: {result['nan_count']}")
        print(f"    Inf count: {result['inf_count']}")

    return result

def check_all_batches(output_dir, data_dir, batches, variant_name):
    results = {}
    all_pass = True
    for b in batches:
        out_path = os.path.join(output_dir, f"output_b{b}.bin")
        ref_path = os.path.join(data_dir, f"reference_b{b}_fp16.bin")
        if not os.path.exists(out_path):
            print(f"  [{variant_name}] b={b}: output not found at {out_path}")
            results[b] = {"status": "SKIP", "error": "output file not found"}
            all_pass = False
            continue
        result = check_correctness(out_path, ref_path, b, verbose=False)
        results[b] = result
        if result["status"] != "PASS":
            all_pass = False
            print(f"  [{variant_name}] b={b}: FAILED ({result['mismatch_count']} mismatches, max_abs={result['max_abs_diff']:.6e})")
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
    result = check_correctness(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    sys.exit(0 if result["status"] == "PASS" else 1)
