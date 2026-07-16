import numpy as np
import os
import json

SEED = 20260715
rng = np.random.default_rng(SEED)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

BOUNDARY_CASES = [
    "all_true",
    "all_false",
    "alternating",
    "random_mask",
    "sparse_true",
    "dense_true",
]


def generate_x(batch, case_name):
    shape = [batch] + SHAPE_TAIL
    total_elems = int(np.prod(shape))
    local_rng = np.random.default_rng(SEED + hash(case_name) % 100000)

    if case_name == "all_true":
        data = np.ones(total_elems, dtype=np.uint8)
    elif case_name == "all_false":
        data = np.zeros(total_elems, dtype=np.uint8)
    elif case_name == "alternating":
        data = np.array([(1 if i % 2 == 0 else 0) for i in range(total_elems)], dtype=np.uint8)
    elif case_name == "random_mask":
        data = local_rng.integers(0, 2, size=total_elems, dtype=np.uint8)
    elif case_name == "sparse_true":
        data = np.zeros(total_elems, dtype=np.uint8)
        num_true = max(1, total_elems // 100)
        indices = local_rng.choice(total_elems, size=num_true, replace=False)
        data[indices] = 1
    elif case_name == "dense_true":
        data = np.ones(total_elems, dtype=np.uint8)
        num_false = max(1, total_elems // 100)
        indices = local_rng.choice(total_elems, size=num_false, replace=False)
        data[indices] = 0
    else:
        raise ValueError(f"Unknown boundary case: {case_name}")

    data = data.reshape(shape)
    return data


def reference_not(x):
    return np.where(x == 0, np.uint8(1), np.uint8(0))


def generate_all():
    for b in BATCHES:
        for case_name in BOUNDARY_CASES:
            x = generate_x(b, case_name)
            y = reference_not(x)

            x_path = os.path.join(DATA_DIR, f"x_b{b}_{case_name}.bin")
            y_path = os.path.join(DATA_DIR, f"reference_b{b}_{case_name}.bin")
            x.tofile(x_path)
            y.tofile(y_path)

    manifest = {
        "operator": "not",
        "seed": SEED,
        "dtype": "bool (uint8)",
        "shape": "[B, 256, 384]",
        "element_count_per_batch": 256 * 384,
        "byte_size_per_input_per_batch": 256 * 384 * 1,
        "batches": BATCHES,
        "boundary_cases": BOUNDARY_CASES,
        "generation_scripts": {
            "inputs": "generation_scripts/generate_inputs.py"
        },
        "input_files": [],
        "reference_files": []
    }

    for b in BATCHES:
        for case_name in BOUNDARY_CASES:
            manifest["input_files"].append({
                "name": f"x_b{b}_{case_name}.bin",
                "batch": b,
                "case": case_name,
                "shape": f"[{b}, 256, 384]",
                "dtype": "bool (uint8)",
                "element_count": b * 256 * 384,
                "byte_size": b * 256 * 384 * 1,
                "seed": SEED,
                "sha256": ""
            })
            manifest["reference_files"].append({
                "name": f"reference_b{b}_{case_name}.bin",
                "batch": b,
                "case": case_name,
                "shape": f"[{b}, 256, 384]",
                "dtype": "bool (uint8)",
                "element_count": b * 256 * 384,
                "byte_size": b * 256 * 384 * 1,
                "seed": SEED,
                "computation": "logical_not(x): uint8 NOT (0->1, nonzero->0)",
                "sha256": ""
            })

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    print(f"Generating NOT data with seed={SEED}")
    generate_all()
    print("All inputs generated successfully.")
