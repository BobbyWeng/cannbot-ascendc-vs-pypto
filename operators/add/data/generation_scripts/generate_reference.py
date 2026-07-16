import numpy as np
import os
import hashlib

SEED = 20260715
BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_fp16_reference(batch):
    """Generate FP16 bitwise reference with strict left-associative order:
    t1 = float16(X1 + X2)
    t2 = float16(t1 + X3)
    Y_ref = float16(t2 + X4)
    """
    x1 = np.fromfile(os.path.join(DATA_DIR, f"x1_b{batch}_fp16.bin"), dtype=np.float16)
    x2 = np.fromfile(os.path.join(DATA_DIR, f"x2_b{batch}_fp16.bin"), dtype=np.float16)
    x3 = np.fromfile(os.path.join(DATA_DIR, f"x3_b{batch}_fp16.bin"), dtype=np.float16)
    x4 = np.fromfile(os.path.join(DATA_DIR, f"x4_b{batch}_fp16.bin"), dtype=np.float16)

    shape = [batch] + SHAPE_TAIL
    x1 = x1.reshape(shape)
    x2 = x2.reshape(shape)
    x3 = x3.reshape(shape)
    x4 = x4.reshape(shape)

    t1 = np.add(x1, x2, dtype=np.float16)
    t2 = np.add(t1, x3, dtype=np.float16)
    y_ref = np.add(t2, x4, dtype=np.float16)

    ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_fp16.bin")
    y_ref.tofile(ref_path)
    return y_ref


def generate_fp32_reference(batch):
    """Generate FP32 high-precision reference:
    Y_ref_fp32 = float32(X1) + float32(X2) + float32(X3) + float32(X4)
    """
    x1 = np.fromfile(os.path.join(DATA_DIR, f"x1_b{batch}_fp16.bin"), dtype=np.float16)
    x2 = np.fromfile(os.path.join(DATA_DIR, f"x2_b{batch}_fp16.bin"), dtype=np.float16)
    x3 = np.fromfile(os.path.join(DATA_DIR, f"x3_b{batch}_fp16.bin"), dtype=np.float16)
    x4 = np.fromfile(os.path.join(DATA_DIR, f"x4_b{batch}_fp16.bin"), dtype=np.float16)

    shape = [batch] + SHAPE_TAIL
    sum32 = x1.astype(np.float32) + x2.astype(np.float32) + x3.astype(np.float32) + x4.astype(np.float32)

    ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_fp32.bin")
    sum32.reshape(shape).tofile(ref_path)
    return sum32


def generate_boundary_references(batch):
    """Generate references for all boundary cases."""
    boundaries = [
        "all_positive", "all_negative", "all_zero",
        "x1_zero_all", "x1_ones_all",
        "large_fp16", "overflow", "underflow",
        "pos_neg_mixed", "small_values"
    ]
    for boundary in boundaries:
        # Check if all 4 boundary inputs exist
        paths = []
        all_exist = True
        for i in range(1, 5):
            p = os.path.join(DATA_DIR, f"x{i}_b{batch}_{boundary}.bin")
            if not os.path.exists(p):
                all_exist = False
                break
            paths.append(p)

        if not all_exist:
            continue

        x1 = np.fromfile(paths[0], dtype=np.float16).reshape([batch] + SHAPE_TAIL)
        x2 = np.fromfile(paths[1], dtype=np.float16).reshape([batch] + SHAPE_TAIL)
        x3 = np.fromfile(paths[2], dtype=np.float16).reshape([batch] + SHAPE_TAIL)
        x4 = np.fromfile(paths[3], dtype=np.float16).reshape([batch] + SHAPE_TAIL)

        t1 = np.add(x1, x2, dtype=np.float16)
        t2 = np.add(t1, x3, dtype=np.float16)
        y_ref = np.add(t2, x4, dtype=np.float16)

        ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_{boundary}.bin")
        y_ref.tofile(ref_path)
        print(f"  b={batch} {boundary}: reference ok")


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def update_manifest():
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        return
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)

    for ref in manifest["reference_files"]:
        name = ref["name"]
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            ref["sha256"] = compute_sha256(path)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest updated with SHA256 hashes.")


if __name__ == "__main__":
    print("Generating reference outputs with fixed computation order:")
    print("  t1 = float16(X1 + X2)")
    print("  t2 = float16(t1 + X3)")
    print("  Y  = float16(t2 + X4)")
    print()

    for b in BATCHES:
        x1_path = os.path.join(DATA_DIR, f"x1_b{b}_fp16.bin")
        if not os.path.exists(x1_path):
            print(f"  b={b}: inputs not found, skipping")
            continue

        ref = generate_fp16_reference(b)
        n_pos0 = int(np.sum(ref == 0.0))
        n_neg0 = int(np.sum(ref == -0.0))
        print(f"  b={b} FP16 ref: ok, zeros={n_pos0}, neg_zeros={n_neg0}")

        generate_fp32_reference(b)
        print(f"  b={b} FP32 ref: ok")

        generate_boundary_references(b)

    update_manifest()
    print("All references generated.")
