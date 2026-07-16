#!/usr/bin/env python3
import os, sys, json
import numpy as np
import torch
import torch_npu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden"))

from transpose_impl import transpose_wrapper
from transpose_golden import transpose_golden

H, W = 256, 384
BATCHES = [1, 2, 4, 8, 16, 32, 64]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_transpose():
    device = torch.npu.current_device()
    if device is None:
        device = 0
        torch.npu.set_device(device)

    all_pass = True
    results = []

    for batch in BATCHES:
        in_path = os.path.join(DATA_DIR, f"input_b{batch}_fp16.bin")
        if not os.path.exists(in_path):
            print(f"  B={batch}: SKIP (input not found)")
            results.append({"batch": batch, "status": "SKIP"})
            continue

        x = torch.from_numpy(np.fromfile(in_path, dtype=np.float16).reshape([batch, H, W]))
        x_npu = x.npu(device)

        try:
            y_pypto = transpose_wrapper(x_npu)
            torch.npu.synchronize(device)
        except Exception as e:
            print(f"  B={batch}: FAIL (runtime error: {e})")
            results.append({"batch": batch, "status": "FAIL", "error": str(e)})
            all_pass = False
            continue

        y_golden = transpose_golden(x_npu)

        # Compare
        out_cpu = y_pypto.cpu()
        ref_cpu = y_golden.cpu()

        shape_match = out_cpu.shape == ref_cpu.shape
        contiguous_ok = out_cpu.is_contiguous()
        expected_shape = torch.Size([batch, W, H])

        if out_cpu.shape != expected_shape:
            print(f"  B={batch}: FAIL (shape: expected {expected_shape}, got {out_cpu.shape})")
            results.append({"batch": batch, "status": "FAIL", "error": f"shape mismatch: {out_cpu.shape}"})
            all_pass = False
            continue

        if not contiguous_ok:
            print(f"  B={batch}: FAIL (not contiguous)")
            results.append({"batch": batch, "status": "FAIL", "error": "not contiguous"})
            all_pass = False
            continue

        bitwise_equal = torch.equal(out_cpu.view(torch.uint16), ref_cpu.view(torch.uint16))

        if bitwise_equal:
            print(f"  B={batch}: [PRECISION_PASS] (bitwise exact, contiguous)")
            results.append({"batch": batch, "status": "PRECISION_PASS", "contiguous": True})
        else:
            mismatch = int((out_cpu.view(torch.uint16) != ref_cpu.view(torch.uint16)).sum().item())
            print(f"  B={batch}: [PRECISION_FAIL] ({mismatch} bitwise mismatches)")
            results.append({"batch": batch, "status": "PRECISION_FAIL", "mismatches": mismatch})
            all_pass = False

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}")

    result_file = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(result_file, "w") as f:
        json.dump({"operator": "transpose", "results": results, "all_pass": all_pass}, f, indent=2)

    return all_pass


if __name__ == "__main__":
    passed = test_transpose()
    sys.exit(0 if passed else 1)
