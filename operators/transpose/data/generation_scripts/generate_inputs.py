import numpy as np
import os

SEED = 20260715
rng = np.random.default_rng(SEED)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
H, W = 256, 384
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_fp16_input(batch, output_path, rng_state=None):
    if rng_state is not None:
        rng = np.random.default_rng(rng_state)
    shape = [batch, H, W]
    total_elems = int(np.prod(shape))

    # Row-identifiable pattern: first rows are monotonic to make transpose visually verifiable
    row_id_elems = min(H * W, 384 * 64)
    row_ids = np.zeros(row_id_elems, dtype=np.float16)
    for i in range(min(H, row_id_elems // W)):
        start = i * W
        end = min(start + W, row_id_elems)
        row_ids[start:end] = np.arange(end - start, dtype=np.float16) + i * 1000.0

    special_count = min(256, total_elems - row_id_elems)
    specials = []
    specials.extend([1.0] * 16)
    specials.extend([-1.0] * 16)
    specials.extend([0.0] * 16)
    specials.extend([-0.0] * 16)
    specials.extend([65504.0] * 16)
    specials.extend([-65504.0] * 16)
    specials.extend([1e-3] * 16)
    specials.extend([-1e-3] * 16)
    specials.extend([float('nan')] * 8)
    specials.extend([float('inf')] * 8)
    specials.extend([float('-inf')] * 8)
    specials.extend([
        0.00006104,  # FP16 min subnormal
        -0.00006104,
        0.00006103515625,  # FP16 smallest normal
        -0.00006103515625,
    ])
    specials.extend(rng.uniform(-10, 10, special_count - len(specials)).tolist())
    specials = specials[:special_count]

    random_count = total_elems - row_id_elems - special_count
    if random_count < 0:
        random_count = 0
    rand_vals = rng.uniform(-100, 100, random_count).tolist() if random_count > 0 else []

    all_vals = np.array(row_ids.tolist() + specials + rand_vals, dtype=np.float16)
    rng.shuffle(all_vals)
    all_vals = all_vals.reshape(shape)

    all_vals.tofile(output_path)
    return all_vals


def generate_boundary_input(batch, output_dir):
    shape = [batch, H, W]
    total_elems = int(np.prod(shape))

    # Row-identifiable monotonic matrix
    data = np.zeros(total_elems, dtype=np.float16)
    for i in range(H):
        row_start = i * W
        data[row_start:row_start + W] = np.arange(W, dtype=np.float16) + i * 1000.0
    data = data.reshape(shape)
    output_path = os.path.join(output_dir, f"input_b{batch}_monotonic.bin")
    data.tofile(output_path)
    return output_path


if __name__ == "__main__":
    print(f"Generating inputs with seed={SEED}")
    for b in BATCHES:
        output_path = os.path.join(DATA_DIR, f"input_b{b}_fp16.bin")
        data = generate_fp16_input(b, output_path, rng_state=SEED + b)
        print(f"  b={b}: shape={list(data.shape)}, min={data.min():.4f}, max={data.max():.4f}, "
              f"nan={int(np.sum(np.isnan(data)))}, inf={int(np.sum(np.isinf(data)))}")

        generate_boundary_input(b, DATA_DIR)

    print("All inputs generated successfully.")
