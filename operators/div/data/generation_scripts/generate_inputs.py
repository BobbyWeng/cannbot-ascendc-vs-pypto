import numpy as np
import os
import hashlib
import json

SEED = 20260715
rng = np.random.default_rng(SEED)

BATCHES = [1, 2, 4, 8, 16, 32]
OPTIONAL_BATCHES = [64]
LOGICAL_TAIL = [3, 4, 256, 256]
KERNEL_TAIL = [12, 256, 256]
X2_LOGICAL_TAIL = [3, 4, 256, 1]
X2_KERNEL_TAIL = [12, 256, 1]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_fp16_x1(batch, output_path, rng_state=None):
    if rng_state is not None:
        rng = np.random.default_rng(rng_state)
    shape = [batch] + KERNEL_TAIL
    total_elems = int(np.prod(shape))

    special_count = min(256, total_elems)
    specials = []
    specials.extend([0.0] * 8)
    specials.extend([-0.0] * 8)
    specials.extend([1e-5] * 4)
    specials.extend([-1e-5] * 4)
    specials.extend([3.0e4] * 4)
    specials.extend([-3.0e4] * 4)
    specials.extend([65504.0] * 4)
    specials.extend([-65504.0] * 4)
    specials.extend(rng.uniform(-4, 4, special_count - len(specials)).tolist())
    specials = specials[:special_count]

    random_count = total_elems - special_count
    rand_vals = rng.uniform(-4, 4, random_count).tolist()

    all_vals = np.array(specials + rand_vals, dtype=np.float16)
    rng.shuffle(all_vals)
    all_vals = all_vals.reshape(shape)

    all_vals.tofile(output_path)

    sha256 = hashlib.sha256()
    sha256.update(all_vals.tobytes())
    return all_vals, sha256.hexdigest()


def generate_fp16_x2(batch, output_path, rng_state=None):
    if rng_state is not None:
        rng = np.random.default_rng(rng_state)
    shape = [batch] + X2_KERNEL_TAIL
    total_elems = int(np.prod(shape))

    specials = [0.25, -0.25, 1.0, -1.0, 4.0, -4.0] * 4
    special_count = len(specials)

    random_count = total_elems - special_count
    half = random_count // 2
    neg_vals = rng.uniform(-4, -0.25, half).tolist()
    pos_vals = rng.uniform(0.25, 4, random_count - half).tolist()

    all_vals = np.array(specials + neg_vals + pos_vals, dtype=np.float16)
    rng.shuffle(all_vals)
    all_vals = all_vals.reshape(shape)

    all_vals.tofile(output_path)

    sha256 = hashlib.sha256()
    sha256.update(all_vals.tobytes())
    return all_vals, sha256.hexdigest()


def generate_special_case(name, batch, output_dir):
    shape = [batch] + KERNEL_TAIL
    x2_shape = [batch] + X2_KERNEL_TAIL
    total_elems = int(np.prod(shape))
    x2_elems = int(np.prod(x2_shape))

    x1_path = os.path.join(output_dir, f"x1_b{batch}_{name}.bin")
    x2_path = os.path.join(output_dir, f"x2_b{batch}_{name}.bin")

    x1 = np.zeros(total_elems, dtype=np.float16)
    x2 = np.zeros(x2_elems, dtype=np.float16)

    if name == "positive_div_pos_zero":
        x1[:] = 2.0
        x2[:] = 0.0
    elif name == "positive_div_neg_zero":
        x1[:] = 2.0
        x2[:] = -0.0
    elif name == "negative_div_pos_zero":
        x1[:] = -2.0
        x2[:] = 0.0
    elif name == "negative_div_neg_zero":
        x1[:] = -2.0
        x2[:] = -0.0
    elif name == "zero_div_zero":
        x1[:] = 0.0
        x2[:] = 0.0
    elif name == "finite_div_inf":
        x1[:] = 2.0
        x2[:] = np.inf
    elif name == "inf_div_finite":
        x1[:] = np.inf
        x2[:] = 2.0
    elif name == "nan_input":
        x1[:] = np.nan
        x2[:] = 2.0
    elif name == "max_fp16_div_finite":
        x1[:] = 65504.0
        x2[:] = 0.5
    elif name == "min_normal_div_finite":
        x1[:] = 6.1035e-5
        x2[:] = 2.0
    elif name == "subnormal_div_finite":
        x1[:] = 6e-8
        x2[:] = 2.0
    elif name == "tiny_nonzero_divisor":
        x1[:] = 2.0
        x2[:] = 1e-4

    x1.tofile(x1_path)
    x2.tofile(x2_path)

    sha1 = hashlib.sha256(x1.tobytes()).hexdigest()
    sha2 = hashlib.sha256(x2.tobytes()).hexdigest()
    return sha1, sha2


if __name__ == "__main__":
    print(f"Generating Div broadcast inputs with seed={SEED}")
    manifest = {
        "operator": "div",
        "seed": SEED,
        "kernel_shape_template": KERNEL_TAIL,
        "x2_kernel_shape_template": X2_KERNEL_TAIL,
        "logical_shape_template": LOGICAL_TAIL,
        "dtype": "float16",
        "seed": SEED,
        "files": []
    }

    for b in BATCHES:
        shape = [b] + KERNEL_TAIL
        x2_shape = [b] + X2_KERNEL_TAIL
        total_elems = int(np.prod(shape))
        x2_elems = int(np.prod(x2_shape))
        byte_size = total_elems * 2
        x2_byte_size = x2_elems * 2

        x1_path = os.path.join(DATA_DIR, f"x1_b{b}_fp16.bin")
        x2_path = os.path.join(DATA_DIR, f"x2_b{b}_fp16.bin")

        x1_data, x1_sha = generate_fp16_x1(b, x1_path, rng_state=SEED + b*2)
        x2_data, x2_sha = generate_fp16_x2(b, x2_path, rng_state=SEED + b*2 + 1)

        x1_range = f"[{x1_data.min():.4f}, {x1_data.max():.4f}]"
        x2_range = f"[{x2_data.min():.4f}, {x2_data.max():.4f}]"

        assert np.all((x2_data >= 0.25) | (x2_data <= -0.25)), f"X2 has values near zero for b={b}"

        manifest["files"].append({
            "batch": b,
            "file": f"x1_b{b}_fp16.bin",
            "shape": shape,
            "dtype": "float16",
            "element_count": total_elems,
            "byte_size": byte_size,
            "seed": SEED + b*2,
            "sha256": x1_sha,
        })
        manifest["files"].append({
            "batch": b,
            "file": f"x2_b{b}_fp16.bin",
            "shape": x2_shape,
            "dtype": "float16",
            "element_count": x2_elems,
            "byte_size": x2_byte_size,
            "seed": SEED + b*2 + 1,
            "sha256": x2_sha,
        })

        print(f"  b={b}: x1 shape={shape}, range={x1_range}, x2 shape={x2_shape}, range={x2_range}")

        for name in [
            "positive_div_pos_zero", "positive_div_neg_zero",
            "negative_div_pos_zero", "negative_div_neg_zero",
            "zero_div_zero", "finite_div_inf", "inf_div_finite",
            "nan_input", "max_fp16_div_finite", "min_normal_div_finite",
            "subnormal_div_finite", "tiny_nonzero_divisor"
        ]:
            sha1, sha2 = generate_special_case(name, b, DATA_DIR)
            manifest["files"].append({
                "batch": b,
                "file": f"x1_b{b}_{name}.bin",
                "boundary": name,
                "dtype": "float16",
                "sha256": sha1,
            })
            manifest["files"].append({
                "batch": b,
                "file": f"x2_b{b}_{name}.bin",
                "boundary": name,
                "dtype": "float16",
                "sha256": sha2,
            })

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")
    print("All Div inputs generated successfully.")
