import numpy as np
import hashlib
import os

OUTPUT_SHAPE_TAIL = [256, 384]

def check_correctness(output_path, reference_path, batch, verbose=True):
    output = np.fromfile(output_path, dtype=np.float16)
    reference = np.fromfile(reference_path, dtype=np.float16)

    total_elements = int(np.prod([batch] + OUTPUT_SHAPE_TAIL))
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

    output_bits = output.view(np.uint16)
    reference_bits = reference.view(np.uint16)

    bitwise_mismatch_mask = output_bits != reference_bits
    bitwise_mismatch_count = int(np.sum(bitwise_mismatch_mask))

    signed_zero_mismatch_mask = (output_bits != reference_bits) & (
        ((output_bits == 0x0000) & (reference_bits == 0x8000)) |
        ((output_bits == 0x8000) & (reference_bits == 0x0000))
    )
    signed_zero_mismatch_count = int(np.sum(signed_zero_mismatch_mask))

    if bitwise_mismatch_count > 0:
        diff = np.abs(output.astype(np.float32) - reference.astype(np.float32))
        max_abs_diff = float(np.max(diff))
        max_rel_diff = float(np.max(diff / (np.abs(reference.astype(np.float32)) + 1e-30)))
    else:
        max_abs_diff = 0.0
        max_rel_diff = 0.0

    bitwise_equal = (bitwise_mismatch_count == 0)

    result = {
        "status": "PASS" if bitwise_equal else "FAIL",
        "total_elements": int(total_elements),
        "checked_elements": int(output.size),
        "bitwise_equal": bitwise_equal,
        "numeric_mismatch_count": bitwise_mismatch_count - signed_zero_mismatch_count,
        "bitwise_mismatch_count": bitwise_mismatch_count,
        "signed_zero_mismatch_count": signed_zero_mismatch_count,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "neg_zero_in_output": int(np.sum(output_bits == 0x8000)),
        "neg_zero_in_reference": int(np.sum(reference_bits == 0x8000)),
    }

    if verbose:
        print(f"  Correctness: {result['status']}")
        print(f"    Elements checked: {result['checked_elements']}")
        print(f"    Bitwise equal: {result['bitwise_equal']}")
        print(f"    Numeric mismatches: {result['numeric_mismatch_count']}")
        print(f"    Bitwise mismatches: {result['bitwise_mismatch_count']}")
        print(f"    Signed zero mismatches: {result['signed_zero_mismatch_count']}")
        print(f"    Max abs diff: {result['max_abs_diff']:.6e}")
        print(f"    Max rel diff: {result['max_rel_diff']:.6e}")
        print(f"    NaN count: {result['nan_count']}")
        print(f"    Inf count: {result['inf_count']}")
        print(f"    -0 in output: {result['neg_zero_in_output']}")
        print(f"    -0 in reference: {result['neg_zero_in_reference']}")

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
            print(f"  [{variant_name}] b={b}: FAILED ({result['bitwise_mismatch_count']} mismatches)")
        else:
            print(f"  [{variant_name}] b={b}: PASS")

    return all_pass, results

def check_boundary(batch, output_dir, data_dir, boundary_name, variant_name):
    out_path = os.path.join(output_dir, f"output_b{batch}_{boundary_name}.bin")
    ref_path = os.path.join(data_dir, f"reference_b{batch}_{boundary_name}.bin")

    if not os.path.exists(out_path):
        return {"status": "SKIP", "error": f"output file not found: {out_path}"}
    if not os.path.exists(ref_path):
        return {"status": "SKIP", "error": f"reference file not found: {ref_path}"}

    result = check_correctness(out_path, ref_path, batch, verbose=False)
    return result

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
