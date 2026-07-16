import numpy as np
import os
import json
import hashlib

SEED = 20260715
rng = np.random.default_rng(SEED)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = [256, 384]
OP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(OP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

INPUT_COMBINATIONS = [
    "false_false",
    "false_true",
    "true_false",
    "true_true",
    "random_mask",
    "sparse",
    "dense",
]


def generate_input_bool(batch, name, output_dir, rng_seed):
    local_rng = np.random.default_rng(rng_seed)
    shape = [batch] + SHAPE_TAIL
    total_elems = int(np.prod(shape))

    if name == "false_false":
        x1 = np.zeros(total_elems, dtype=np.uint8)
        x2 = np.zeros(total_elems, dtype=np.uint8)
        ref = np.zeros(total_elems, dtype=np.uint8)
    elif name == "false_true":
        x1 = np.zeros(total_elems, dtype=np.uint8)
        x2 = np.ones(total_elems, dtype=np.uint8)
        ref = np.ones(total_elems, dtype=np.uint8)
    elif name == "true_false":
        x1 = np.ones(total_elems, dtype=np.uint8)
        x2 = np.zeros(total_elems, dtype=np.uint8)
        ref = np.ones(total_elems, dtype=np.uint8)
    elif name == "true_true":
        x1 = np.ones(total_elems, dtype=np.uint8)
        x2 = np.ones(total_elems, dtype=np.uint8)
        ref = np.ones(total_elems, dtype=np.uint8)
    elif name == "random_mask":
        x1 = local_rng.integers(0, 2, total_elems, dtype=np.uint8)
        x2 = local_rng.integers(0, 2, total_elems, dtype=np.uint8)
        ref = np.logical_or(x1, x2).astype(np.uint8)
    elif name == "sparse":
        x1 = np.zeros(total_elems, dtype=np.uint8)
        x2 = np.zeros(total_elems, dtype=np.uint8)
        sparse_count = max(1, total_elems // 100)
        sparse_indices = local_rng.choice(total_elems, sparse_count, replace=False)
        x2.flat[sparse_indices] = 1
        ref = np.logical_or(x1, x2).astype(np.uint8)
    elif name == "dense":
        x1 = np.ones(total_elems, dtype=np.uint8)
        x2 = np.ones(total_elems, dtype=np.uint8)
        zero_count = max(1, total_elems // 50)
        zero_indices = local_rng.choice(total_elems, zero_count, replace=False)
        x1.flat[zero_indices] = 0
        ref = np.logical_or(x1, x2).astype(np.uint8)
    else:
        raise ValueError(f"Unknown combination: {name}")

    x1 = x1.reshape(shape)
    x2 = x2.reshape(shape)
    ref = ref.reshape(shape)

    x1_path = os.path.join(output_dir, f"x1_b{batch}_{name}.bin")
    x2_path = os.path.join(output_dir, f"x2_b{batch}_{name}.bin")
    ref_path = os.path.join(output_dir, f"reference_b{batch}_{name}.bin")

    x1.tofile(x1_path)
    x2.tofile(x2_path)
    ref.tofile(ref_path)

    return x1, x2, ref


def generate_manifest():
    manifest = {
        "operator": "or",
        "seed": SEED,
        "dtype": "bool",
        "storage_dtype": "uint8",
        "shape": "[B, 256, 384]",
        "element_count_per_batch": 256 * 384,
        "byte_size_per_input_per_batch": 256 * 384,
        "batches": BATCHES,
        "input_combinations": INPUT_COMBINATIONS,
        "generation_scripts": {
            "inputs": "generation_scripts/generate_inputs.py",
            "correctness": "generation_scripts/correctness.py"
        },
        "input_files": [],
        "reference_files": []
    }

    for b in BATCHES:
        for variant in INPUT_COMBINATIONS:
            manifest["input_files"].append({
                "name": f"x1_b{b}_{variant}.bin",
                "batch": b,
                "variant": variant,
                "shape": f"[{b}, 256, 384]",
                "dtype": "bool",
                "element_count": b * 256 * 384,
                "byte_size": b * 256 * 384,
                "seed": SEED,
                "sha256": ""
            })
            manifest["input_files"].append({
                "name": f"x2_b{b}_{variant}.bin",
                "batch": b,
                "variant": variant,
                "shape": f"[{b}, 256, 384]",
                "dtype": "bool",
                "element_count": b * 256 * 384,
                "byte_size": b * 256 * 384,
                "seed": SEED,
                "sha256": ""
            })
            manifest["reference_files"].append({
                "name": f"reference_b{b}_{variant}.bin",
                "batch": b,
                "variant": variant,
                "shape": f"[{b}, 256, 384]",
                "dtype": "bool",
                "element_count": b * 256 * 384,
                "byte_size": b * 256 * 384,
                "seed": SEED,
                "computation": "torch.logical_or(x1, x2).to(torch.uint8)",
                "sha256": ""
            })
    return manifest


def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha.update(chunk)
    return sha.hexdigest()


if __name__ == "__main__":
    print(f"Generating OR inputs with seed={SEED}")
    manifest = generate_manifest()

    for b in BATCHES:
        for variant in INPUT_COMBINATIONS:
            x1, x2, ref = generate_input_bool(b, variant, DATA_DIR, SEED + b * 10 + hash(variant) % 1000)

            for prefix, tensor, entry_list in [
                ("x1", x1, manifest["input_files"]),
                ("x2", x2, manifest["input_files"]),
                ("reference", ref, manifest["reference_files"]),
            ]:
                fname = f"{prefix}_b{b}_{variant}.bin"
                fpath = os.path.join(DATA_DIR, fname)
                sha = compute_sha256(fpath)
                for mf in entry_list:
                    if mf["name"] == fname:
                        mf["sha256"] = sha
                        break

            true_count = int(x1.sum()) + int(x2.sum())
            print(f"  b={b} {variant}: x1_true={int(x1.sum())}, x2_true={int(x2.sum())}, ref_true={int(ref.sum())}")

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")
    print("All OR inputs generated successfully.")
