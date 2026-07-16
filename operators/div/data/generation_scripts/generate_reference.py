import numpy as np
import hashlib
import os
import json

SEED = 20260715
BATCHES = [1, 2, 4, 8, 16, 32]
KERNEL_TAIL = [12, 256, 256]
X2_KERNEL_TAIL = [12, 256, 1]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_reference(batch):
    x1_path = os.path.join(DATA_DIR, f"x1_b{batch}_fp16.bin")
    x2_path = os.path.join(DATA_DIR, f"x2_b{batch}_fp16.bin")
    ref_fp16_path = os.path.join(DATA_DIR, f"reference_b{batch}_fp16.bin")
    ref_fp32_path = os.path.join(DATA_DIR, f"reference_b{batch}_fp32.bin")

    if not os.path.exists(x1_path) or not os.path.exists(x2_path):
        return None

    x1 = np.fromfile(x1_path, dtype=np.float16).reshape([batch] + KERNEL_TAIL)
    x2_raw = np.fromfile(x2_path, dtype=np.float16).reshape([batch] + X2_KERNEL_TAIL)

    x2_broadcast = np.broadcast_to(x2_raw, [batch] + KERNEL_TAIL)

    ref_fp32 = x1.astype(np.float32) / x2_broadcast.astype(np.float32)
    ref_fp32.tofile(ref_fp32_path)

    ref_fp16 = ref_fp32.astype(np.float16)
    ref_fp16.tofile(ref_fp16_path)

    sha256_fp16 = hashlib.sha256(ref_fp16.tobytes()).hexdigest()
    sha256_fp32 = hashlib.sha256(ref_fp32.tobytes()).hexdigest()

    n_pos0 = int(np.sum(ref_fp16 == 0.0))
    n_neg0 = int(np.sum(ref_fp16 == -0.0))
    n_nan = int(np.sum(np.isnan(ref_fp16)))
    n_inf = int(np.sum(np.isinf(ref_fp16)))

    print(f"  b={batch}: ref_fp16 ok, +0={n_pos0}, -0={n_neg0}, NaN={n_nan}, Inf={n_inf}")
    return [
        {
            "batch": batch,
            "file": f"reference_b{batch}_fp16.bin",
            "dtype": "float16",
            "sha256": sha256_fp16,
            "pos_zero_count": int(n_pos0),
            "neg_zero_count": int(n_neg0),
            "nan_count": int(n_nan),
            "inf_count": int(n_inf),
        },
        {
            "batch": batch,
            "file": f"reference_b{batch}_fp32.bin",
            "dtype": "float32",
            "sha256": sha256_fp32,
        }
    ]


def generate_special_reference(batch, name):
    x1_path = os.path.join(DATA_DIR, f"x1_b{batch}_{name}.bin")
    x2_path = os.path.join(DATA_DIR, f"x2_b{batch}_{name}.bin")
    ref_fp16_path = os.path.join(DATA_DIR, f"reference_b{batch}_{name}.bin")

    if not os.path.exists(x1_path) or not os.path.exists(x2_path):
        return None

    x1 = np.fromfile(x1_path, dtype=np.float16).reshape([batch] + KERNEL_TAIL)
    x2_raw = np.fromfile(x2_path, dtype=np.float16).reshape([batch] + X2_KERNEL_TAIL)

    x2_broadcast = np.broadcast_to(x2_raw, [batch] + KERNEL_TAIL)

    ref = (x1.astype(np.float32) / x2_broadcast.astype(np.float32)).astype(np.float16)
    ref.tofile(ref_fp16_path)

    sha256 = hashlib.sha256(ref.tobytes()).hexdigest()
    print(f"  b={batch} {name}: reference ok")
    return {
        "batch": batch,
        "boundary": name,
        "file": f"reference_b{batch}_{name}.bin",
        "sha256": sha256,
    }


if __name__ == "__main__":
    print("Generating Div broadcast reference outputs...")
    manifest_entries = []
    for b in BATCHES:
        entries = generate_reference(b)
        if entries:
            manifest_entries.extend(entries)

    for b in BATCHES:
        for name in [
            "positive_div_pos_zero", "positive_div_neg_zero",
            "negative_div_pos_zero", "negative_div_neg_zero",
            "zero_div_zero", "finite_div_inf", "inf_div_finite",
            "nan_input", "max_fp16_div_finite", "min_normal_div_finite",
            "subnormal_div_finite", "tiny_nonzero_divisor"
        ]:
            entry = generate_special_reference(b, name)
            if entry:
                manifest_entries.append(entry)

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        manifest["references"] = manifest_entries
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest updated at {manifest_path}")

    print("All references generated.")
