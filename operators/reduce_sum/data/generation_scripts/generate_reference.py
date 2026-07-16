import numpy as np
import os
import json

SEED = 20260715

BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

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


def reference_fp16_accum(x_fp16):
    """Simulate FP16 accumulation: sum in FP16, chained."""
    rows = x_fp16.reshape(-1, 384)
    result = np.zeros(rows.shape[0], dtype=np.float16)
    for i in range(rows.shape[0]):
        s = np.float16(0.0)
        for j in range(384):
            s = np.float16(float(s) + float(rows[i, j]))
        result[i] = s
    result = result.reshape(x_fp16.shape[:-1])
    return result


def generate_all():
    for b in BATCHES:
        for case_name in COVERAGE_CASES:
            x_path = os.path.join(DATA_DIR, f"x_b{b}_{case_name}.bin")
            if not os.path.exists(x_path):
                print(f"  WARNING: {x_path} not found, skipping.")
                continue
            x = np.fromfile(x_path, dtype=np.float16).reshape([b] + SHAPE_TAIL)

            ref_fp32 = np.sum(x, axis=-1, dtype=np.float32).astype(np.float16)
            ref_fp16 = reference_fp16_accum(x)

            ref_fp32_path = os.path.join(DATA_DIR, f"reference_b{b}_{case_name}_fp32_accum.bin")
            ref_fp16_path = os.path.join(DATA_DIR, f"reference_b{b}_{case_name}_fp16_path.bin")

            ref_fp32.tofile(ref_fp32_path)
            ref_fp16.tofile(ref_fp16_path)

            ref_torch = np.sum(x.astype(np.float32), axis=-1).astype(np.float16)
            torch_path = os.path.join(DATA_DIR, f"reference_b{b}_{case_name}_fp16.bin")
            ref_torch.tofile(torch_path)

            if ref_fp32.shape == ref_fp16.shape:
                max_diff = np.max(np.abs(ref_fp32.astype(np.float32) - ref_fp16.astype(np.float32)))
            else:
                max_diff = float('nan')
                print(f"  WARNING: shape mismatch fp32={ref_fp32.shape} fp16={ref_fp16.shape}")
            print(f"  B={b} {case_name}: max_diff(fp32_accum vs fp16_path)={max_diff:.6f}")

    print("All references generated.")


if __name__ == "__main__":
    print("Generating ReduceSum references (FP32 accum + FP16 path + torch).")
    generate_all()
