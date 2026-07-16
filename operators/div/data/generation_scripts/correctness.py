import numpy as np
import hashlib
import os

KERNEL_TAIL = [12, 256, 256]
X2_KERNEL_TAIL = [12, 256, 1]


def ulp_diff_fp16(a, b):
    a_bits = a.view(np.uint16)
    b_bits = b.view(np.uint16)
    return np.abs(a_bits.astype(np.int32) - b_bits.astype(np.int32))


def check_correctness(output_path, reference_fp16_path, reference_fp32_path, batch, variant_name="", verbose=True):
    output = np.fromfile(output_path, dtype=np.float16)
    ref_fp16 = np.fromfile(reference_fp16_path, dtype=np.float16)
    ref_fp32 = np.fromfile(reference_fp32_path, dtype=np.float32)

    total_elements = int(np.prod([batch] + KERNEL_TAIL))

    if output.size != ref_fp16.size:
        return {"status": "FAIL", "error": f"Size mismatch: output={output.size}, ref={ref_fp16.size}"}

    output_fp32 = output.astype(np.float32)

    nan_output = np.isnan(output)
    nan_ref = np.isnan(ref_fp16)
    inf_output = np.isinf(output)
    inf_ref = np.isinf(ref_fp16)

    finite_mask = ~nan_output & ~inf_output & ~nan_ref & ~inf_ref
    finite_elements = int(np.sum(finite_mask))

    abs_diff = np.full(output.shape, np.nan, dtype=np.float32)
    rel_diff = np.full(output.shape, np.nan, dtype=np.float32)
    ulp_diff = np.full(output.shape, -1, dtype=np.int32)

    abs_diff[finite_mask] = np.abs(output_fp32[finite_mask] - ref_fp32[finite_mask])
    rel_diff[finite_mask] = abs_diff[finite_mask] / (np.abs(ref_fp32[finite_mask]) + 1e-30)
    ulp_diff[finite_mask] = ulp_diff_fp16(output[finite_mask], ref_fp16[finite_mask])

    max_abs_diff = float(np.nanmax(abs_diff)) if finite_elements > 0 else 0.0
    max_rel_diff = float(np.nanmax(rel_diff)) if finite_elements > 0 else 0.0
    max_ulp = int(np.max(ulp_diff)) if finite_elements > 0 else 0

    mismatch_count = int(np.sum(finite_mask & (abs_diff > 0.001)))
    rel_mismatch_count = int(np.sum(finite_mask & (rel_diff > 0.001)))

    output_bits = output.view(np.uint16)
    ref_bits = ref_fp16.view(np.uint16)

    bitwise_mismatch_count = int(np.sum(output_bits != ref_bits))

    signed_zero_mask = (output_bits != ref_bits) & (
        ((output_bits == 0x0000) & (ref_bits == 0x8000)) |
        ((output_bits == 0x8000) & (ref_bits == 0x0000))
    )
    signed_zero_mismatch_count = int(np.sum(signed_zero_mask))

    nan_class_mismatch = int(np.sum(nan_output != nan_ref))
    inf_class_mismatch = int(np.sum(inf_output != inf_ref))
    inf_sign_mismatch = int(np.sum(
        inf_output & inf_ref & (np.signbit(output) != np.signbit(ref_fp16))
    ))

    overflow_count = int(np.sum(inf_output & ~inf_ref))
    underflow_count = int(np.sum(
        (output == 0.0) & (ref_fp16 != 0.0) & ~np.isnan(ref_fp16) & ~np.isinf(ref_fp16)
    ))

    atol = 0.001
    rtol = 0.001
    within_tolerance = np.all(abs_diff[finite_mask] <= atol) & np.all(rel_diff[finite_mask] <= rtol) if finite_elements > 0 else True

    passed = within_tolerance and nan_class_mismatch == 0

    result = {
        "variant": variant_name,
        "batch": batch,
        "status": "PASS" if passed else "FAIL",
        "total_elements": int(output.size),
        "checked_elements": finite_elements,
        "finite_mismatch_count": int(np.sum(finite_mask & (abs_diff > atol))),
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "max_ulp": max_ulp,
        "bitwise_mismatch_count": bitwise_mismatch_count,
        "signed_zero_mismatch_count": signed_zero_mismatch_count,
        "nan_classification_mismatch": nan_class_mismatch,
        "inf_classification_mismatch": inf_class_mismatch,
        "inf_sign_mismatch": inf_sign_mismatch,
        "overflow_count": overflow_count,
        "underflow_count": underflow_count,
        "neg_zero_in_output": int(np.sum(output_bits == 0x8000)),
        "neg_zero_in_reference": int(np.sum(ref_bits == 0x8000)),
    }

    if verbose:
        print(f"  [{variant_name}] b={batch}: {result['status']}")
        print(f"    Total elements: {result['total_elements']}")
        print(f"    Finite checked: {result['checked_elements']}")
        print(f"    Max abs diff: {result['max_abs_diff']:.6e}")
        print(f"    Max rel diff: {result['max_rel_diff']:.6e}")
        print(f"    Max ULP: {result['max_ulp']}")
        print(f"    Finite mismatches (atol={atol}): {result['finite_mismatch_count']}")
        print(f"    NaN class mismatch: {result['nan_classification_mismatch']}")
        print(f"    Inf class mismatch: {result['inf_classification_mismatch']}")
        print(f"    Inf sign mismatch: {result['inf_sign_mismatch']}")
        print(f"    Signed zero mismatch: {result['signed_zero_mismatch_count']}")
        print(f"    Overflow count: {result['overflow_count']}")
        print(f"    Underflow count: {result['underflow_count']}")

    return result


def check_all_batches(output_dir, data_dir, batches, variant_name):
    results = {}
    all_pass = True
    for b in batches:
        out_path = os.path.join(output_dir, f"output_b{b}.bin")
        ref_fp16_path = os.path.join(data_dir, f"reference_b{b}_fp16.bin")
        ref_fp32_path = os.path.join(data_dir, f"reference_b{b}_fp32.bin")
        if not os.path.exists(out_path):
            print(f"  [{variant_name}] b={b}: output not found")
            results[b] = {"status": "SKIP", "error": "output not found"}
            all_pass = False
            continue
        result = check_correctness(out_path, ref_fp16_path, ref_fp32_path, b, variant_name, verbose=False)
        results[b] = result
        if result["status"] != "PASS":
            all_pass = False
            print(f"  [{variant_name}] b={b}: FAILED (finite_mismatch={result['finite_mismatch_count']}, NaN_mismatch={result['nan_classification_mismatch']})")
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
        print("Usage: correctness.py <output_path> <reference_fp16_path> <reference_fp32_path> <batch>")
        sys.exit(1)
    result = check_correctness(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    sys.exit(0 if result["status"] == "PASS" else 1)
