import numpy as np
import os

SEED = 20260715
rng = np.random.default_rng(SEED)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
NORM_SHAPE = [256, 32]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_fp16_input(batch, output_path, rng_state=None):
    if rng_state is not None:
        rng = np.random.default_rng(rng_state)
    shape = [batch] + NORM_SHAPE
    total_elems = int(np.prod(shape))

    special_count = min(256, total_elems)
    specials = []
    specials.extend([1.0] * 16)
    specials.extend([-1.0] * 16)
    specials.extend([0.0] * 16)
    specials.extend([-0.0] * 16)
    specials.extend([65504.0] * 16)
    specials.extend([-65504.0] * 16)
    specials.extend([1e-3] * 16)
    specials.extend([-1e-3] * 16)
    specials.extend(rng.uniform(-10, 10, special_count - len(specials)).tolist())
    specials = specials[:special_count]

    random_count = total_elems - special_count
    rand_vals = rng.uniform(-100, 100, random_count).tolist()

    all_vals = np.array(specials + rand_vals, dtype=np.float16)
    rng.shuffle(all_vals)
    all_vals = all_vals.reshape(shape)

    all_vals.tofile(output_path)
    return all_vals

def generate_boundary_input(batch, name, output_dir):
    shape = [batch] + NORM_SHAPE
    total_elems = int(np.prod(shape))

    if name == "all_positive":
        data = np.ones(total_elems, dtype=np.float16) * 3.14
    elif name == "all_negative":
        data = np.ones(total_elems, dtype=np.float16) * (-3.14)
    elif name == "all_zero":
        data = np.zeros(total_elems, dtype=np.float16)
    elif name == "constant":
        data = np.ones(total_elems, dtype=np.float16) * 42.0
    else:
        raise ValueError(f"Unknown boundary case: {name}")

    data = data.reshape(shape)
    output_path = os.path.join(output_dir, f"input_b{batch}_{name}.bin")
    data.tofile(output_path)
    return output_path

if __name__ == "__main__":
    print(f"Generating inputs with seed={SEED}")

    w_path = os.path.join(DATA_DIR, "weight_fp16.bin")
    w = rng.uniform(-2, 2, int(np.prod(NORM_SHAPE))).astype(np.float16).reshape(NORM_SHAPE)
    w.tofile(w_path)
    print(f"  weight: shape={NORM_SHAPE}, min={w.min():.4f}, max={w.max():.4f}")

    b_path = os.path.join(DATA_DIR, "bias_fp16.bin")
    b = rng.uniform(-1, 1, int(np.prod(NORM_SHAPE))).astype(np.float16).reshape(NORM_SHAPE)
    b.tofile(b_path)
    print(f"  bias: shape={NORM_SHAPE}, min={b.min():.4f}, max={b.max():.4f}")

    for b in BATCHES:
        output_path = os.path.join(DATA_DIR, f"input_b{b}_fp16.bin")
        data = generate_fp16_input(b, output_path, rng_state=SEED + b)
        print(f"  b={b}: shape={list(data.shape)}, min={data.min():.4f}, max={data.max():.4f}")

        for boundary in ["all_positive", "all_negative", "all_zero", "constant"]:
            generate_boundary_input(b, boundary, DATA_DIR)

    print("All inputs generated successfully.")
