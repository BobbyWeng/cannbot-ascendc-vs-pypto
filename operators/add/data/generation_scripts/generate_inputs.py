import numpy as np
import os
import hashlib

SEED = 20260715
rng = np.random.default_rng(SEED)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_input(batch, input_idx, output_path, rng_seed):
    local_rng = np.random.default_rng(rng_seed + input_idx * 1000 + batch)
    shape = [batch] + SHAPE_TAIL
    total_elems = int(np.prod(shape))

    special_count = min(512, total_elems)
    specials = []
    specials.extend([1.0] * 32)
    specials.extend([-1.0] * 32)
    specials.extend([0.0] * 32)
    specials.extend([-0.0] * 32)
    specials.extend([65504.0] * 16)
    specials.extend([-65504.0] * 16)
    specials.extend([1e-3] * 16)
    specials.extend([-1e-3] * 16)
    specials.extend([1e-5] * 16)
    specials.extend([-1e-5] * 16)
    specials.extend([64512.0] * 16)
    specials.extend([-64512.0] * 16)
    specials.extend([6.0e-8] * 16)
    specials.extend([-6.0e-8] * 16)
    specials.extend(local_rng.uniform(-10, 10, special_count - len(specials)).tolist())
    specials = specials[:special_count]

    random_count = total_elems - special_count
    rand_vals = local_rng.uniform(-100, 100, random_count).tolist()

    all_vals = np.array(specials + rand_vals, dtype=np.float16)
    local_rng.shuffle(all_vals)
    all_vals = all_vals.reshape(shape)

    all_vals.tofile(output_path)
    return all_vals


def generate_boundary_input(batch, input_idx, name, output_dir):
    shape = [batch] + SHAPE_TAIL
    total_elems = int(np.prod(shape))

    if name == "all_positive":
        data = np.ones(total_elems, dtype=np.float16) * 1.5
    elif name == "all_negative":
        data = np.ones(total_elems, dtype=np.float16) * -1.5
    elif name == "all_zero":
        data = np.zeros(total_elems, dtype=np.float16)
    elif name == "x1_zero_all":
        data = np.zeros(total_elems, dtype=np.float16)
    elif name == "x1_ones_all":
        data = np.ones(total_elems, dtype=np.float16)
    elif name == "large_fp16":
        data = np.ones(total_elems, dtype=np.float16) * 60000.0
    elif name == "overflow":
        data = np.ones(total_elems, dtype=np.float16) * 32000.0
    elif name == "underflow":
        data = np.ones(total_elems, dtype=np.float16) * 6e-8
    elif name == "pos_neg_mixed":
        data = np.array([(1.0 if i % 2 == 0 else -1.0) for i in range(total_elems)], dtype=np.float16)
    elif name == "small_values":
        data = np.ones(total_elems, dtype=np.float16) * 1.0e-4
    else:
        raise ValueError(f"Unknown boundary case: {name}")

    data = data.reshape(shape)
    output_path = os.path.join(output_dir, f"x{input_idx}_b{batch}_{name}.bin")
    data.tofile(output_path)
    return output_path


def generate_manifest():
    manifest = {
        "operator": "add",
        "seed": SEED,
        "dtype": "float16",
        "shape": "[B, 256, 384]",
        "element_count_per_batch": 256 * 384,
        "byte_size_per_input_per_batch": 256 * 384 * 2,
        "batches": BATCHES,
        "boundary_cases": [
            "all_positive", "all_negative", "all_zero",
            "x1_zero_all", "x1_ones_all",
            "large_fp16", "overflow", "underflow",
            "pos_neg_mixed", "small_values"
        ],
        "generation_scripts": {
            "inputs": "generation_scripts/generate_inputs.py",
            "reference": "generation_scripts/generate_reference.py",
            "correctness": "generation_scripts/correctness.py"
        },
        "input_files": [],
        "reference_files": []
    }
    for b in BATCHES:
        for i in range(1, 5):
            manifest["input_files"].append({
                "name": f"x{i}_b{b}_fp16.bin",
                "batch": b,
                "input_index": i,
                "shape": f"[{b}, 256, 384]",
                "dtype": "float16",
                "element_count": b * 256 * 384,
                "byte_size": b * 256 * 384 * 2,
                "seed": SEED,
                "sha256": ""
            })
        manifest["reference_files"].append({
            "name": f"reference_b{b}_fp16.bin",
            "batch": b,
            "shape": f"[{b}, 256, 384]",
            "dtype": "float16",
            "element_count": b * 256 * 384,
            "byte_size": b * 256 * 384 * 2,
            "seed": SEED,
            "computation_order": "((X1+X2)+X3)+X4 in FP16",
            "sha256": ""
        })
        manifest["reference_files"].append({
            "name": f"reference_b{b}_fp32.bin",
            "batch": b,
            "shape": f"[{b}, 256, 384]",
            "dtype": "float32",
            "element_count": b * 256 * 384,
            "byte_size": b * 256 * 384 * 4,
            "computation_order": "float32(X1)+float32(X2)+float32(X3)+float32(X4)",
            "sha256": ""
        })
    return manifest


if __name__ == "__main__":
    print(f"Generating inputs with seed={SEED}")
    manifest = generate_manifest()

    for b in BATCHES:
        for i in range(1, 5):
            output_path = os.path.join(DATA_DIR, f"x{i}_b{b}_fp16.bin")
            data = generate_input(b, i, output_path, SEED + b * 10 + i)
            sha = hashlib.sha256()
            with open(output_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    sha.update(chunk)
            m = None
            for mf in manifest["input_files"]:
                if mf["name"] == f"x{i}_b{b}_fp16.bin":
                    mf["sha256"] = sha.hexdigest()
                    break
            print(f"  b={b} x{i}: shape={list(data.shape)}, min={data.min():.4f}, max={data.max():.4f}, "
                  f"pos0={int(np.sum(data == 0.0))}, neg0={int(np.sum(data == -0.0))}, sha256={sha.hexdigest()[:16]}...")

        for boundary in manifest["boundary_cases"]:
            for i in range(1, 5):
                generate_boundary_input(b, i, boundary, DATA_DIR)

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        import json
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")
    print("All inputs generated successfully.")
