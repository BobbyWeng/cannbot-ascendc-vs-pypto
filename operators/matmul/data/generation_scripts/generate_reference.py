#!/usr/bin/env python3
"""Generate FP16 and FP32 reference outputs for MatMul.

References computed as:
  Y[b,h,:,:] = A[b,h,:,:] @ B[b,h,:,:]

FP32 reference uses FP32 accumulation.
FP16 reference uses torch.matmul with FP16 inputs (native FP16 accumulation).
"""
import os, sys, json, hashlib
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCHES = [1, 2, 4, 8, 16, 32]
HEADS = 12
M, N, K = 256, 32, 256


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def generate_reference(batch, case):
    a_path = os.path.join(DATA_DIR, f"A_b{batch}_{case}.bin")
    b_path = os.path.join(DATA_DIR, f"B_b{batch}_{case}.bin")

    A = np.fromfile(a_path, dtype=np.float16).reshape(batch, HEADS, M, K)
    B_mat = np.fromfile(b_path, dtype=np.float16).reshape(batch, HEADS, K, N)

    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B_mat)

    # FP16 reference (native torch.matmul)
    ref_fp16 = torch.matmul(A_t.half(), B_t.half()).half()

    # FP32 accumulation reference
    ref_fp32 = torch.matmul(A_t.float(), B_t.float()).half()

    ref_fp16_path = os.path.join(DATA_DIR, f"reference_b{batch}_{case}_fp16.bin")
    ref_fp32_path = os.path.join(DATA_DIR, f"reference_b{batch}_{case}_fp32.bin")

    ref_fp16.numpy().tofile(ref_fp16_path)
    ref_fp32.numpy().tofile(ref_fp32_path)

    return ref_fp16_path, ref_fp32_path


def compute_tolerance_stats(batch):
    """Compute tolerance statistics from FP16 vs FP32 reference for perf case."""
    a_path = os.path.join(DATA_DIR, f"A_b{batch}_perf_fp16.bin")
    b_path = os.path.join(DATA_DIR, f"B_b{batch}_perf_fp16.bin")

    A = np.fromfile(a_path, dtype=np.float16).reshape(batch, HEADS, M, K)
    B_mat = np.fromfile(b_path, dtype=np.float16).reshape(batch, HEADS, K, N)

    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B_mat)

    ref_fp16 = torch.matmul(A_t.half(), B_t.half())
    ref_fp32 = torch.matmul(A_t.float(), B_t.float())

    abs_diff = (ref_fp16.half().float() - ref_fp32.half().float()).abs()
    rel_diff = abs_diff / (ref_fp32.half().float().abs() + 1e-12)

    return {
        "batch": batch,
        "max_abs_diff": float(abs_diff.max().item()),
        "mean_abs_diff": float(abs_diff.mean().item()),
        "max_rel_diff": float(rel_diff.max().item()),
        "mean_rel_diff": float(rel_diff.mean().item()),
        "rmse": float(torch.sqrt((abs_diff ** 2).mean()).item()),
    }


def main():
    performance_cases = ["perf_fp16"]
    special_cases = [
        "zeros", "ones", "identity", "sparse", "small_values",
        "mixed_signs", "cancellation", "max_safe", "underflow",
        "overflow_risk", "nan", "inf",
    ]
    tolerance_stats = []

    for batch in BATCHES:
        for case in performance_cases + special_cases:
            ref_fp16_path, ref_fp32_path = generate_reference(batch, case)
            print(f"B={batch} case={case}: FP16 ref -> {ref_fp16_path}, FP32 ref -> {ref_fp32_path}")

        stats = compute_tolerance_stats(batch)
        tolerance_stats.append(stats)
        print(f"B={batch}: max_abs_diff={stats['max_abs_diff']:.6f}, max_rel_diff={stats['max_rel_diff']:.6f}")

    # Compute overall tolerance recommendation
    max_abs = max(s["max_abs_diff"] for s in tolerance_stats)
    max_rel = max(s["max_rel_diff"] for s in tolerance_stats)
    print(f"\nTolerance Recommendation:")
    print(f"  max_abs_diff across all batches (FP16 vs FP32): {max_abs:.6f}")
    print(f"  max_rel_diff across all batches: {max_rel:.6f}")
    print(f"  Recommended gate: atol={max_abs*2:.6f}, rtol={max_rel*2:.6f}")

    tolerance_path = os.path.join(DATA_DIR, "tolerance_recommendation.json")
    with open(tolerance_path, "w") as f:
        json.dump({
            "tolerance_stats": tolerance_stats,
            "recommended_atol": round(max_abs * 2, 6),
            "recommended_rtol": round(max_rel * 2, 6),
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "note": "FP16 MatMul accumulation differs from FP32 accumulation. Tolerance must account for this."
        }, f, indent=2)
    print(f"Tolerance recommendation saved to {tolerance_path}")


if __name__ == "__main__":
    main()
