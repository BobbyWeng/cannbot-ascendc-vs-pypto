import numpy as np
import os

SEED = 20260715
BATCHES = [1, 2, 4, 8, 16, 32, 64]
H, W = 256, 384
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_reference(batch, input_path, ref_path):
    data = np.fromfile(input_path, dtype=np.float16)
    shape = [batch, H, W]
    data = data.reshape(shape)
    # Transpose [B,256,384] -> [B,384,256], materialize contiguous
    ref = np.ascontiguousarray(data.transpose(0, 2, 1))
    ref.tofile(ref_path)
    return ref


if __name__ == "__main__":
    print("Generating reference outputs...")
    for b in BATCHES:
        input_path = os.path.join(DATA_DIR, f"input_b{b}_fp16.bin")
        ref_path = os.path.join(DATA_DIR, f"reference_b{b}_fp16.bin")
        if os.path.exists(input_path):
            ref = generate_reference(b, input_path, ref_path)
            shape = [b, W, H]
            print(f"  b={b}: reference ok, shape={ref.shape}")
        else:
            print(f"  b={b}: input not found, skipping")

    # Also generate reference for monotonic inputs
    for b in BATCHES:
        input_path = os.path.join(DATA_DIR, f"input_b{b}_monotonic.bin")
        ref_path = os.path.join(DATA_DIR, f"reference_b{b}_monotonic.bin")
        if os.path.exists(input_path):
            generate_reference(b, input_path, ref_path)
            print(f"  b={b} monotonic: reference ok")

    print("All references generated.")
