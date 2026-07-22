import numpy as np
import os

SEED = 20260715
EPS = 1e-5
BATCHES = [1, 2, 4, 8, 16, 32, 64]
NORM_SHAPE = [256, 32]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def layernorm_ref(x, weight, bias, eps=EPS):
    orig_dtype = x.dtype
    x_f32 = x.astype(np.float32)
    w_f32 = weight.astype(np.float32)
    b_f32 = bias.astype(np.float32)

    last_dim_size = x_f32.shape[-1]
    mean = np.mean(x_f32, axis=-1, keepdims=True)
    x_centered = x_f32 - mean
    var = np.mean(x_centered ** 2, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    normalized = x_centered * inv_std
    y = normalized * w_f32 + b_f32
    return y.astype(orig_dtype)

def generate_reference(batch, input_path, ref_path, weight, bias):
    data = np.fromfile(input_path, dtype=np.float16)
    shape = [batch] + NORM_SHAPE
    data = data.reshape(shape)
    ref = layernorm_ref(data, weight, bias)
    ref.tofile(ref_path)
    return ref

if __name__ == "__main__":
    print("Generating reference outputs...")

    w_path = os.path.join(DATA_DIR, "weight_fp16.bin")
    b_path = os.path.join(DATA_DIR, "bias_fp16.bin")
    weight = np.fromfile(w_path, dtype=np.float16).reshape(NORM_SHAPE)
    bias = np.fromfile(b_path, dtype=np.float16).reshape(NORM_SHAPE)

    for b in BATCHES:
        input_path = os.path.join(DATA_DIR, f"input_b{b}_fp16.bin")
        ref_path = os.path.join(DATA_DIR, f"reference_b{b}_fp16.bin")
        if os.path.exists(input_path):
            ref = generate_reference(b, input_path, ref_path, weight, bias)
            print(f"  b={b}: reference ok, min={ref.min():.4f}, max={ref.max():.4f}")
        else:
            print(f"  b={b}: input not found, skipping")

    for b in BATCHES:
        for boundary in ["all_positive", "all_negative", "all_zero", "constant"]:
            input_path = os.path.join(DATA_DIR, f"input_b{b}_{boundary}.bin")
            ref_path = os.path.join(DATA_DIR, f"reference_b{b}_{boundary}.bin")
            if os.path.exists(input_path):
                generate_reference(b, input_path, ref_path, weight, bias)
                print(f"  b={b} {boundary}: reference ok")

    print("All references generated.")
