import numpy as np
import os
import json

SEED = 20260715
rng = np.random.default_rng(SEED)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

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

SPECIAL_CASES = ["nan", "inf"]


def generate_x(batch, case_name):
    shape = [batch] + SHAPE_TAIL
    total_elems = int(np.prod(shape))
    local_rng = np.random.default_rng(SEED + hash(case_name) % 100000)

    if case_name == "random_finite":
        data = local_rng.uniform(-1.0, 1.0, size=total_elems).astype(np.float16)
    elif case_name == "all_zero":
        data = np.zeros(total_elems, dtype=np.float16)
    elif case_name == "all_one":
        data = np.ones(total_elems, dtype=np.float16)
    elif case_name == "pos_neg_cancel":
        half = total_elems // 2
        pos = local_rng.uniform(0.1, 1.0, size=half).astype(np.float16)
        neg = local_rng.uniform(-1.0, -0.1, size=half).astype(np.float16)
        combined = np.concatenate([pos, neg])
        data = combined[:total_elems]
    elif case_name == "small_values":
        data = local_rng.uniform(-1e-4, 1e-4, size=total_elems).astype(np.float16)
    elif case_name == "large_values":
        data = local_rng.uniform(-1e4, 1e4, size=total_elems).astype(np.float16)
    elif case_name == "overflow_risk":
        data = np.full(total_elems, 65504.0, dtype=np.float16)
        num_neg = max(1, total_elems // 100)
        indices = local_rng.choice(total_elems, size=num_neg, replace=False)
        data[indices] = -65504.0
    elif case_name == "underflow_risk":
        data = local_rng.uniform(-1e-5, 1e-5, size=total_elems).astype(np.float16)
    elif case_name == "nan":
        data = np.full(total_elems, np.nan, dtype=np.float16)
    elif case_name == "inf":
        half = total_elems // 2
        data = np.concatenate([
            np.full(half, np.inf, dtype=np.float16),
            np.full(total_elems - half, -np.inf, dtype=np.float16),
        ])

    data = data.reshape(shape).astype(np.float16)
    return data


def generate_all():
    for b in BATCHES:
        for case_name in COVERAGE_CASES:
            x = generate_x(b, case_name)
            x_path = os.path.join(DATA_DIR, f"x_b{b}_{case_name}.bin")
            x.tofile(x_path)

    manifest = {
        "operator": "reduce_sum",
        "seed": SEED,
        "dtype": "float16",
        "shape": "[B, 256, 384]",
        "element_count_per_batch": 256 * 384,
        "byte_size_per_input_per_batch": 256 * 384 * 2,
        "batches": BATCHES,
        "coverage_cases": COVERAGE_CASES,
        "special_cases": SPECIAL_CASES,
        "generation_scripts": {
            "inputs": "generation_scripts/generate_inputs.py"
        },
        "input_files": [],
    }

    for b in BATCHES:
        for case_name in COVERAGE_CASES:
            manifest["input_files"].append({
                "name": f"x_b{b}_{case_name}.bin",
                "batch": b,
                "case": case_name,
                "shape": f"[{b}, 256, 384]",
                "dtype": "float16",
                "element_count": b * 256 * 384,
                "byte_size": b * 256 * 384 * 2,
                "seed": SEED,
                "sha256": ""
            })

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    print(f"Generating ReduceSum input data with seed={SEED}")
    generate_all()
    print("All inputs generated successfully.")
