import numpy as np
import os
import hashlib
import json

SEED = 20260715
rng = np.random.default_rng(SEED)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TEMPLATE = [3, 4, 256, 32]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_fp16_input(batch, variant, output_path, rng_state=None):
    if rng_state is not None:
        rng = np.random.default_rng(rng_state)
    shape = [batch] + SHAPE_TEMPLATE
    total_elems = int(np.prod(shape))

    special_count = min(256, total_elems)
    specials = []
    # +0 and -0
    specials.extend([0.0] * 8)
    specials.extend([-0.0] * 8)
    # small finite values
    specials.extend([1e-5] * 4)
    specials.extend([-1e-5] * 4)
    # large finite values
    specials.extend([1e4] * 4)
    specials.extend([-1e4] * 4)
    # overflow candidates
    specials.extend([3.0e4] * 4)
    specials.extend([-3.0e4] * 4)
    # underflow candidates
    specials.extend([1e-8] * 4)
    specials.extend([-1e-8] * 4)
    # values near FP16 max
    specials.extend([65504.0] * 4)
    specials.extend([-65504.0] * 4)
    # NaN/Inf (not generated — FP16 multiply can overflow to Inf)
    # normal random values
    specials.extend(rng.uniform(-100, 100, special_count - len(specials)).tolist())
    specials = specials[:special_count]

    random_count = total_elems - special_count
    rand_vals = rng.uniform(-100, 100, random_count).tolist()

    all_vals = np.array(specials + rand_vals, dtype=np.float16)
    rng.shuffle(all_vals)
    all_vals = all_vals.reshape(shape)

    all_vals.tofile(output_path)
    
    sha256 = hashlib.sha256()
    sha256.update(all_vals.tobytes())
    return all_vals, sha256.hexdigest()

def generate_boundary(batch, name, output_dir):
    shape = [batch] + SHAPE_TEMPLATE
    total_elems = int(np.prod(shape))
    seed_offset = abs(hash(name)) % 10000
    
    if name == "all_positive":
        x1 = np.ones(total_elems, dtype=np.float16) * 2.5
        x2 = np.ones(total_elems, dtype=np.float16) * 1.5
    elif name == "all_negative":
        x1 = np.ones(total_elems, dtype=np.float16) * -2.5
        x2 = np.ones(total_elems, dtype=np.float16) * -1.5
    elif name == "mixed_sign":
        x1 = np.ones(total_elems, dtype=np.float16) * 3.0
        x2 = np.ones(total_elems, dtype=np.float16) * -2.0
    elif name == "x1_zero":
        x1 = np.zeros(total_elems, dtype=np.float16)
        x2 = np.ones(total_elems, dtype=np.float16) * 2.0
    elif name == "x2_zero":
        x1 = np.ones(total_elems, dtype=np.float16) * 2.0
        x2 = np.zeros(total_elems, dtype=np.float16)
    elif name == "both_zero":
        x1 = np.zeros(total_elems, dtype=np.float16)
        x2 = np.zeros(total_elems, dtype=np.float16)
    elif name == "x1_one":
        x1 = np.ones(total_elems, dtype=np.float16)
        x2 = rng.uniform(-10, 10, total_elems).astype(np.float16)
    elif name == "x2_one":
        x1 = rng.uniform(-10, 10, total_elems).astype(np.float16)
        x2 = np.ones(total_elems, dtype=np.float16)
    elif name == "overflow":
        x1 = np.ones(total_elems, dtype=np.float16) * 256.0
        x2 = np.ones(total_elems, dtype=np.float16) * 256.0
    elif name == "underflow":
        x1 = np.ones(total_elems, dtype=np.float16) * 1e-4
        x2 = np.ones(total_elems, dtype=np.float16) * 1e-4
    elif name == "pos_x_neg":
        x1 = np.ones(total_elems, dtype=np.float16) * 3.0
        x2 = np.ones(total_elems, dtype=np.float16) * -4.0
    elif name == "neg_x_neg":
        x1 = np.ones(total_elems, dtype=np.float16) * -3.0
        x2 = np.ones(total_elems, dtype=np.float16) * -4.0
    else:
        raise ValueError(f"Unknown boundary: {name}")

    x1 = x1.reshape(shape)
    x2 = x2.reshape(shape)
    
    sha1 = hashlib.sha256(x1.tobytes()).hexdigest()
    sha2 = hashlib.sha256(x2.tobytes()).hexdigest()
    
    x1.tofile(os.path.join(output_dir, f"x1_b{batch}_{name}.bin"))
    x2.tofile(os.path.join(output_dir, f"x2_b{batch}_{name}.bin"))
    
    return sha1, sha2

if __name__ == "__main__":
    print(f"Generating Mul inputs with seed={SEED}")
    manifest = {
        "operator": "mul",
        "seed": SEED,
        "shape_template": SHAPE_TEMPLATE,
        "dtype": "float16",
        "files": []
    }
    
    for b in BATCHES:
        shape = [b] + SHAPE_TEMPLATE
        total_elems = int(np.prod(shape))
        byte_size = total_elems * 2
        
        x1_path = os.path.join(DATA_DIR, f"x1_b{b}_fp16.bin")
        x2_path = os.path.join(DATA_DIR, f"x2_b{b}_fp16.bin")
        
        x1_data, x1_sha = generate_fp16_input(b, "x1", x1_path, rng_state=SEED + b*2)
        x2_data, x2_sha = generate_fp16_input(b, "x2", x2_path, rng_state=SEED + b*2 + 1)
        
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
            "shape": shape,
            "dtype": "float16",
            "element_count": total_elems,
            "byte_size": byte_size,
            "seed": SEED + b*2 + 1,
            "sha256": x2_sha,
        })
        
        print(f"  b={b}: shape={shape}, x1 range=[{x1_data.min():.4f}, {x1_data.max():.4f}], x2 range=[{x2_data.min():.4f}, {x2_data.max():.4f}]")
        
        for boundary in ["all_positive", "all_negative", "mixed_sign",
                          "x1_zero", "x2_zero", "both_zero",
                          "x1_one", "x2_one", "overflow", "underflow",
                          "pos_x_neg", "neg_x_neg"]:
            sha1, sha2 = generate_boundary(b, boundary, DATA_DIR)
            manifest["files"].append({
                "batch": b,
                "file": f"x1_b{b}_{boundary}.bin",
                "boundary": boundary,
                "dtype": "float16",
                "sha256": sha1,
            })
            manifest["files"].append({
                "batch": b,
                "file": f"x2_b{b}_{boundary}.bin",
                "boundary": boundary,
                "dtype": "float16",
                "sha256": sha2,
            })
    
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")
    print("All Mul inputs generated successfully.")
