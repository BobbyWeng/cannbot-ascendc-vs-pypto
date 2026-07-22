#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_DIR)
import torch
import torch_npu
from common.correctness import check_correctness

SHAPE_TAIL = [256, 32]
NORM_SHAPE = [32]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def load_batch_data(batch, device_id):
    x_path = os.path.join(DATA_DIR, f"input_b{batch}_fp16.bin")
    ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_fp16.bin")
    w_path = os.path.join(DATA_DIR, "weight_fp16.bin")
    b_path = os.path.join(DATA_DIR, "bias_fp16.bin")
    shape = [batch] + SHAPE_TAIL

    if not os.path.exists(x_path):
        return None, None, None, None, f"input file not found: {x_path}"
    if not os.path.exists(ref_path):
        return None, None, None, None, f"reference file not found: {ref_path}"

    x = torch.from_numpy(np.fromfile(x_path, dtype=np.float16).reshape(shape))
    ref = torch.from_numpy(np.fromfile(ref_path, dtype=np.float16).reshape(shape))
    w = torch.from_numpy(np.fromfile(w_path, dtype=np.float16).reshape(-1))
    b = torch.from_numpy(np.fromfile(b_path, dtype=np.float16).reshape(-1))
    return x.npu(device_id), w.npu(device_id), b.npu(device_id), ref.npu(device_id), None

def run_correctness(batch, device_id):
    torch.npu.set_device(device_id)
    x_npu, w_npu, b_npu, ref_npu, err = load_batch_data(batch, device_id)
    if err:
        return {"batch": batch, "status": "SKIP", "error": err}

    eps = 1e-5
    output = torch.nn.functional.layer_norm(x_npu, NORM_SHAPE, weight=w_npu, bias=b_npu, eps=eps)
    torch.npu.synchronize(device_id)

    result = check_correctness(output, ref_npu, rtol=0.001, atol=0.01, require_bitwise=False, label=f"B={batch}")
    result["batch"] = batch
    result["shape"] = [batch] + SHAPE_TAIL
    result["dtype"] = "float16"
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline correctness for LayerNorm")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    all_pass = True
    results = []
    for b in batches:
        result = run_correctness(b, args.device)
        results.append(result)
        status = result["status"]
        if status == "PASS":
            print(f"  B={b}: PASS")
        elif status == "SKIP":
            print(f"  B={b}: SKIP ({result.get('error', '')})")
        else:
            print(f"  B={b}: FAIL (mismatches={result.get('numeric_mismatch_count', '?')}, max_abs={result.get('max_abs_diff', '?'):.6e})")
            all_pass = False

    output = {
        "operator": "layernorm",
        "variant": "torch",
        "results": results,
        "all_pass": all_pass,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.exit(0 if all_pass else 1)
