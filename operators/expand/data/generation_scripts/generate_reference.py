import numpy as np
import os

SEED = 20260715
BATCHES = [1, 2, 4, 8, 16, 32, 64]
OUTPUT_SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_reference(batch, input_path, ref_path):
    data = np.fromfile(input_path, dtype=np.float16)
    shape = [batch] + [256, 1]
    data = data.reshape(shape)

    # Expand: Y[b,i,j] = X[b,i,0] — duplicate last dim from 1 to 384
    ref = np.broadcast_to(data, (batch, 256, 384)).copy()
    ref.tofile(ref_path)
    return ref

if __name__ == "__main__":
    print("Generating reference outputs...")
    for b in BATCHES:
        input_path = os.path.join(DATA_DIR, f"input_b{b}_fp16.bin")
        ref_path = os.path.join(DATA_DIR, f"reference_b{b}_fp16.bin")
        if os.path.exists(input_path):
            ref = generate_reference(b, input_path, ref_path)
            print(f"  b={b}: reference ok, shape={list(ref.shape)}")
        else:
            print(f"  b={b}: input not found, skipping")

    for b in BATCHES:
        for boundary in ["all_positive", "all_negative", "all_zero", "alternating_sign"]:
            input_path = os.path.join(DATA_DIR, f"input_b{b}_{boundary}.bin")
            ref_path = os.path.join(DATA_DIR, f"reference_b{b}_{boundary}.bin")
            if os.path.exists(input_path):
                generate_reference(b, input_path, ref_path)
                print(f"  b={b} {boundary}: reference ok")

    print("All references generated.")
