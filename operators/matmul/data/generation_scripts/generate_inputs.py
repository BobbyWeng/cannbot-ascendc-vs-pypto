#!/usr/bin/env python3
"""Generate MatMul inputs for all batch sizes and test cases.

Generates A [B,12,256,256] and B [B,12,256,32] FP16 inputs.
Performance inputs use deterministic seed, finite values, no NaN/Inf.
Special-value inputs cover edge cases for correctness verification.
"""
import os, sys, json, hashlib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN_VERSION = "matmul_v1"
SEED = 20260716
BATCHES = [1, 2, 4, 8, 16, 32]
HEADS = 12
M, N, K = 256, 32, 256

rng = np.random.default_rng(SEED)


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def save_bin(data, path):
    data.astype(np.float16).tofile(path)


def generate_perf_inputs(batch):
    """Generate performance inputs: finite, deterministic, no NaN/Inf."""
    shape_a = (batch, HEADS, M, K)
    shape_b = (batch, HEADS, K, N)
    A = rng.uniform(-1.0, 1.0, shape_a).astype(np.float16)
    B = rng.uniform(-1.0, 1.0, shape_b).astype(np.float16)
    return A, B


def generate_special_inputs(batch, case_name):
    """Generate special-value test inputs."""
    shape_a = (batch, HEADS, M, K)
    shape_b = (batch, HEADS, K, N)

    if case_name == "zeros":
        A = np.zeros(shape_a, dtype=np.float16)
        B = np.zeros(shape_b, dtype=np.float16)
    elif case_name == "ones":
        A = np.ones(shape_a, dtype=np.float16)
        B = np.ones(shape_b, dtype=np.float16)
    elif case_name == "identity":
        A = np.zeros(shape_a, dtype=np.float16)
        for b in range(batch):
            for h in range(HEADS):
                np.fill_diagonal(A[b, h], 1.0)
        B = np.ones(shape_b, dtype=np.float16)
    elif case_name == "sparse":
        A = rng.uniform(-1.0, 1.0, shape_a).astype(np.float16)
        mask = rng.random(shape_a) > 0.1
        A[mask] = 0.0
        B = rng.uniform(-1.0, 1.0, shape_b).astype(np.float16)
        mask = rng.random(shape_b) > 0.1
        B[mask] = 0.0
    elif case_name == "small_values":
        A = rng.uniform(-1e-5, 1e-5, shape_a).astype(np.float16)
        B = rng.uniform(-1e-5, 1e-5, shape_b).astype(np.float16)
    elif case_name == "mixed_signs":
        A = rng.uniform(-2.0, 2.0, shape_a).astype(np.float16)
        A[np.abs(A) < 0.5] = -A[np.abs(A) < 0.5]
        B = rng.uniform(-2.0, 2.0, shape_b).astype(np.float16)
        B[np.abs(B) < 0.5] = -B[np.abs(B) < 0.5]
    elif case_name == "cancellation":
        A = np.ones(shape_a, dtype=np.float16) * 2.0
        half_k = K // 2
        A[..., :half_k] = -2.0
        B = np.ones(shape_b, dtype=np.float16)
    elif case_name == "max_safe":
        A = np.full(shape_a, 65504.0, dtype=np.float16) * 0.5
        B = np.full(shape_b, 65504.0, dtype=np.float16) * 0.5
    elif case_name == "underflow":
        A = rng.uniform(1e-8, 1e-6, shape_a).astype(np.float16)
        B = rng.uniform(1e-8, 1e-6, shape_b).astype(np.float16)
    elif case_name == "overflow_risk":
        A = np.full(shape_a, 65504.0, dtype=np.float16) * 0.9
        B = np.full(shape_b, 65504.0, dtype=np.float16)
    elif case_name == "nan":
        A = rng.uniform(-1.0, 1.0, shape_a).astype(np.float16)
        A[0, 0, 0, 0] = np.float16(np.nan)
        B = rng.uniform(-1.0, 1.0, shape_b).astype(np.float16)
    elif case_name == "inf":
        A = rng.uniform(-1.0, 1.0, shape_a).astype(np.float16)
        A[0, 0, 0, 0] = np.float16(np.inf)
        B = rng.uniform(-1.0, 1.0, shape_b).astype(np.float16)
    else:
        raise ValueError(f"Unknown case: {case_name}")

    return A, B


def main():
    performance_cases = ["fp16"]
    special_cases = [
        "zeros", "ones", "identity", "sparse", "small_values",
        "mixed_signs", "cancellation", "max_safe", "underflow",
        "overflow_risk", "nan", "inf",
    ]
    manifest = {
        "version": GEN_VERSION,
        "seed": SEED,
        "shape": {"B": "variable", "heads": HEADS, "M": M, "K": K, "N": N},
        "batches": BATCHES,
        "dtype": "float16",
        "inputs": [],
        "generation_scripts": ["generate_inputs.py", "generate_reference.py"],
    }

    for batch in BATCHES:
        # Performance input
        A, B_mat = generate_perf_inputs(batch)
        a_path = os.path.join(DATA_DIR, f"A_b{batch}_perf_fp16.bin")
        b_path = os.path.join(DATA_DIR, f"B_b{batch}_perf_fp16.bin")
        save_bin(A, a_path)
        save_bin(B_mat, b_path)
        manifest["inputs"].append({
            "case": "perf", "batch": batch,
            "A": a_path, "B": b_path,
            "A_sha256": _hash_file(a_path),
            "B_sha256": _hash_file(b_path),
        })

        # Special-value inputs
        for case in special_cases:
            A, B_mat = generate_special_inputs(batch, case)
            a_path = os.path.join(DATA_DIR, f"A_b{batch}_{case}.bin")
            b_path = os.path.join(DATA_DIR, f"B_b{batch}_{case}.bin")
            save_bin(A, a_path)
            save_bin(B_mat, b_path)
            manifest["inputs"].append({
                "case": case, "batch": batch,
                "A": a_path, "B": b_path,
                "A_sha256": _hash_file(a_path),
                "B_sha256": _hash_file(b_path),
            })

    man_path = os.path.join(DATA_DIR, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {man_path}")
    print(f"Generated {len(manifest['inputs'])} input pairs")


if __name__ == "__main__":
    main()
