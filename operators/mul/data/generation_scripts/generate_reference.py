import numpy as np
import hashlib
import os
import json

SEED = 20260715
BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TEMPLATE = [3, 4, 256, 32]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_reference(batch):
    x1_path = os.path.join(DATA_DIR, f"x1_b{batch}_fp16.bin")
    x2_path = os.path.join(DATA_DIR, f"x2_b{batch}_fp16.bin")
    ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_fp16.bin")
    
    if not os.path.exists(x1_path) or not os.path.exists(x2_path):
        return False
    
    x1 = np.fromfile(x1_path, dtype=np.float16).reshape([batch] + SHAPE_TEMPLATE)
    x2 = np.fromfile(x2_path, dtype=np.float16).reshape([batch] + SHAPE_TEMPLATE)
    
    # Golden: Y_ref = (X1.float32 * X2.float32).to(float16)
    ref = (x1.astype(np.float32) * x2.astype(np.float32)).astype(np.float16)
    ref.tofile(ref_path)
    
    sha256 = hashlib.sha256(ref.tobytes()).hexdigest()
    
    n_pos0 = int(np.sum(ref == 0.0))
    n_neg0 = int(np.sum(ref == -0.0))
    n_nan = int(np.sum(np.isnan(ref)))
    n_inf = int(np.sum(np.isinf(ref)))
    
    print(f"  b={batch}: reference ok, +0={n_pos0}, -0={n_neg0}, NaN={n_nan}, Inf={n_inf}")
    return {
        "batch": batch,
        "file": f"reference_b{batch}_fp16.bin",
        "sha256": sha256,
        "pos_zero_count": int(n_pos0),
        "neg_zero_count": int(n_neg0),
    }

def generate_boundary_reference(batch, name):
    x1_path = os.path.join(DATA_DIR, f"x1_b{batch}_{name}.bin")
    x2_path = os.path.join(DATA_DIR, f"x2_b{batch}_{name}.bin")
    ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_{name}.bin")
    
    if not os.path.exists(x1_path) or not os.path.exists(x2_path):
        return None
    
    x1 = np.fromfile(x1_path, dtype=np.float16).reshape([batch] + SHAPE_TEMPLATE)
    x2 = np.fromfile(x2_path, dtype=np.float16).reshape([batch] + SHAPE_TEMPLATE)
    
    ref = (x1.astype(np.float32) * x2.astype(np.float32)).astype(np.float16)
    ref.tofile(ref_path)
    
    sha256 = hashlib.sha256(ref.tobytes()).hexdigest()
    print(f"  b={batch} {name}: reference ok")
    return {
        "batch": batch,
        "boundary": name,
        "file": f"reference_b{batch}_{name}.bin",
        "sha256": sha256,
    }

if __name__ == "__main__":
    print("Generating Mul reference outputs...")
    manifest_entries = []
    for b in BATCHES:
        entry = generate_reference(b)
        if entry:
            manifest_entries.append(entry)

    for b in BATCHES:
        for boundary in ["all_positive", "all_negative", "mixed_sign",
                          "x1_zero", "x2_zero", "both_zero",
                          "x1_one", "x2_one", "overflow", "underflow",
                          "pos_x_neg", "neg_x_neg"]:
            entry = generate_boundary_reference(b, boundary)
            if entry:
                manifest_entries.append(entry)

    # Update manifest with reference entries
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        manifest["references"] = manifest_entries
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest updated at {manifest_path}")

    print("All references generated.")
