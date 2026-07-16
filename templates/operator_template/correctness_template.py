#!/usr/bin/env python3
"""Correctness verification template for {{ operator_name }}.
Fill in at the marked location.
"""
import os, sys, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_DIR, "common"))
from common.correctness import check_correctness
import torch, torch_npu

SHAPE_TAIL = {{ shape_tail | default('[12, 256, 32]') }}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ====== OP-SPECIFIC: Implement the operator ======
def op_impl(x):
    return torch.{{ operator_name | default('relu') }}(x)
# =================================================

def verify(batch, device=0):
    input_path = os.path.join(DATA_DIR, f"input_b{batch}_fp16.bin")
    ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_fp16.bin")
    x = torch.from_numpy(np.fromfile(input_path, dtype=np.float16).reshape([batch] + list(SHAPE_TAIL)))
    ref = torch.from_numpy(np.fromfile(ref_path, dtype=np.float16).reshape([batch] + list(SHAPE_TAIL)))
    x_npu = x.npu()
    ref_npu = ref.npu()
    y = op_impl(x_npu)
    return check_correctness(y.cpu(), ref, require_bitwise=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="1,2,4,8,16,32,64")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    batches = [int(b) for b in args.batch.split(",")]
    results = []
    for batch in batches:
        r = verify(batch, args.device)
        results.append({"batch": batch, **r})
        print(f"B={batch}: {r['status']} (numeric_mismatch={r['numeric_mismatch_count']})")
    out = os.path.join(os.path.dirname(__file__), "correctness_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
