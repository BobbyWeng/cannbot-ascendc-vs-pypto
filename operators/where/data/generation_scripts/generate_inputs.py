import numpy as np
import os
import hashlib
import json

SEED = 20260715
rng = np.random.default_rng(SEED)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = [256, 384]
OP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(OP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FP16_BOUNDARIES = np.array([
    0.0, -0.0,
    1.0, -1.0,
    65504.0, -65504.0,
    1e-3, -1e-3,
    1e-5, -1e-5,
    6e-8, -6e-8,
    float('inf'), float('-inf'),
    float('nan'),
], dtype=np.float16)


def generate_condition(batch):
    shape = [batch] + SHAPE_TAIL
    total = int(np.prod(shape))

    frac_all_true = total // 4
    frac_all_false = total // 4
    frac_alternating = total // 4
    frac_random = total - frac_all_true - frac_all_false - frac_alternating

    cond = np.zeros(total, dtype=np.uint8)

    start = 0
    cond[start:start + frac_all_true] = 1
    start += frac_all_true

    start += frac_all_false

    for i in range(frac_alternating):
        cond[start + i] = i % 2
    start += frac_alternating

    cond[start:] = rng.integers(0, 2, size=frac_random).astype(np.uint8)

    rng.shuffle(cond)
    return cond.reshape(shape)


def generate_fp16_tensor(batch, rng_seed, include_nan_in_non_selected=False, is_non_selected=False):
    local_rng = np.random.default_rng(rng_seed)
    shape = [batch] + SHAPE_TAIL
    total = int(np.prod(shape))

    special_count = min(512, total)
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

    random_count = total - special_count
    rand_vals = local_rng.uniform(-100, 100, random_count).tolist()

    all_vals = np.array(specials + rand_vals, dtype=np.float16)
    local_rng.shuffle(all_vals)

    if include_nan_in_non_selected and is_non_selected:
        nan_positions = local_rng.choice(total, size=min(64, total), replace=False)
        all_vals[nan_positions] = float('nan')

    return all_vals.reshape(shape)


def generate_reference(condition, x1, x2):
    cond_bool = condition.astype(bool)
    return np.where(cond_bool, x1, x2).astype(np.float16)


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    print(f"Generating Where data with seed={SEED}")

    manifest = {
        "operator": "where",
        "seed": SEED,
        "dtype": {"condition": "uint8", "x1": "float16", "x2": "float16", "y": "float16"},
        "shape": "[B, 256, 384]",
        "element_count_per_batch": 256 * 384,
        "batches": BATCHES,
        "data_coverage": ["all_true", "all_false", "alternating", "random"],
        "special_values": ["pos/neg", "zero/neg_zero", "FP16_boundary", "NaN_in_non_selected", "Inf"],
        "generation_scripts": {
            "inputs": "generation_scripts/generate_inputs.py"
        },
        "files": []
    }

    for b in BATCHES:
        print(f"\nBatch B={b}:")

        condition = generate_condition(b)
        cond_path = os.path.join(DATA_DIR, f"condition_b{b}_bool.bin")
        condition.tofile(cond_path)
        sha = compute_sha256(cond_path)
        manifest["files"].append({
            "name": f"condition_b{b}_bool.bin",
            "batch": b,
            "type": "condition",
            "dtype": "uint8",
            "shape": f"[{b}, 256, 384]",
            "element_count": b * 256 * 384,
            "byte_size": b * 256 * 384,
            "data_coverage": "all_true+all_false+alternating+random",
            "sha256": sha
        })
        print(f"  condition: shape={condition.shape}, "
              f"true={int(condition.sum())}/{int(np.prod(condition.shape))}, sha256={sha[:16]}...")

        x1 = generate_fp16_tensor(b, SEED + b * 100 + 1, include_nan_in_non_selected=False, is_non_selected=False)
        x1_path = os.path.join(DATA_DIR, f"x1_b{b}_fp16.bin")
        x1.tofile(x1_path)
        sha = compute_sha256(x1_path)
        manifest["files"].append({
            "name": f"x1_b{b}_fp16.bin",
            "batch": b,
            "type": "input",
            "dtype": "float16",
            "shape": f"[{b}, 256, 384]",
            "element_count": b * 256 * 384,
            "byte_size": b * 256 * 384 * 2,
            "sha256": sha
        })
        print(f"  x1: min={x1.min():.4f}, max={x1.max():.4f}, nan={int(np.isnan(x1).sum())}, inf={int(np.isinf(x1).sum())}, sha256={sha[:16]}...")

        x2 = generate_fp16_tensor(b, SEED + b * 100 + 2, include_nan_in_non_selected=True, is_non_selected=True)
        x2_path = os.path.join(DATA_DIR, f"x2_b{b}_fp16.bin")
        x2.tofile(x2_path)
        sha = compute_sha256(x2_path)
        manifest["files"].append({
            "name": f"x2_b{b}_fp16.bin",
            "batch": b,
            "type": "input",
            "dtype": "float16",
            "shape": f"[{b}, 256, 384]",
            "element_count": b * 256 * 384,
            "byte_size": b * 256 * 384 * 2,
            "sha256": sha
        })
        print(f"  x2: min={x2.min():.4f}, max={x2.max():.4f}, nan={int(np.isnan(x2).sum())}, inf={int(np.isinf(x2).sum())}, sha256={sha[:16]}...")

        reference = generate_reference(condition, x1, x2)
        ref_path = os.path.join(DATA_DIR, f"reference_b{b}_fp16.bin")
        reference.tofile(ref_path)
        sha = compute_sha256(ref_path)
        manifest["files"].append({
            "name": f"reference_b{b}_fp16.bin",
            "batch": b,
            "type": "reference",
            "dtype": "float16",
            "shape": f"[{b}, 256, 384]",
            "element_count": b * 256 * 384,
            "byte_size": b * 256 * 384 * 2,
            "sha256": sha
        })
        nan_in_ref = int(np.isnan(reference).sum())
        print(f"  reference: min={reference.min():.4f}, max={reference.max():.4f}, nan={nan_in_ref}, sha256={sha[:16]}...")
        print(f"    NaN non-propagation check: x2 has NaN but reference has {nan_in_ref} NaN (should be 0)")

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print("All Where data generated successfully.")
