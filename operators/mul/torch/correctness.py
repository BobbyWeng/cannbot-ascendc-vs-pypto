#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
import torch
import torch_npu
from common.correctness import check_correctness

SHAPE_TAIL = [3, 4, 256, 32]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_batch_data(batch, device_id):
    x1_path = os.path.join(DATA_DIR, f"x1_b{batch}_fp16.bin")
    x2_path = os.path.join(DATA_DIR, f"x2_b{batch}_fp16.bin")
    ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_fp16.bin")
    shape = [batch] + SHAPE_TAIL

    if not os.path.exists(x1_path):
        return None, None, None, f"x1 file not found: {x1_path}"
    if not os.path.exists(x2_path):
        return None, None, None, f"x2 file not found: {x2_path}"
    if not os.path.exists(ref_path):
        return None, None, None, f"reference file not found: {ref_path}"

    x1 = torch.from_numpy(np.fromfile(x1_path, dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(x2_path, dtype=np.float16).reshape(shape))
    ref = torch.from_numpy(np.fromfile(ref_path, dtype=np.float16).reshape(shape))
    return x1.npu(device_id), x2.npu(device_id), ref.npu(device_id), None


def run_correctness(batch, device_id):
    torch.npu.set_device(device_id)
    x1_npu, x2_npu, ref_npu, err = load_batch_data(batch, device_id)
    if err:
        return {"batch": batch, "status": "SKIP", "error": err}

    output = torch.mul(x1_npu, x2_npu)
    torch.npu.synchronize(device_id)

    result = check_correctness(output, ref_npu, rtol=0, atol=0, require_bitwise=True, label=f"B={batch}")
    result["batch"] = batch
    result["shape"] = [batch] + SHAPE_TAIL
    result["dtype"] = "float16"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline correctness for Mul")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes")
    parser.add_argument("--device", type=int, default=0,
                        help="NPU device ID")
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
            print(f"  B={b}: FAIL (mismatches={result.get('bitwise_mismatch_count', '?')})")
            all_pass = False

    output = {
        "operator": "mul",
        "variant": "torch",
        "results": results,
        "all_pass": all_pass,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.exit(0 if all_pass else 1)
