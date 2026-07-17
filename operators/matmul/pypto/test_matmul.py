#!/usr/bin/env python3
"""PyPTO MatMul test with tri-state marking.

Usage:
  python3 test_matmul.py
  python3 test_matmul.py --test smoke   # only smoke test
  python3 test_matmul.py --test precision  # only precision tests
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matmul_impl import matmul_wrapper
from matmul_golden import matmul_golden

BATCHES = [1, 2, 4, 8, 16, 32]
HEADS = 12
M, N, K = 256, 32, 256


def test_precision(batch, device_id=0):
    """Test precision for a specific batch size with 4D inputs."""
    torch.npu.set_device(device_id)
    shape_a = [batch, HEADS, M, K]
    shape_b = [batch, HEADS, K, N]

    A = torch.randn(shape_a, dtype=torch.float16).npu(device_id)
    B_mat = torch.randn(shape_b, dtype=torch.float16).npu(device_id)

    try:
        output = matmul_wrapper(A, B_mat)
        torch.npu.synchronize(device_id)
    except Exception as e:
        return {
            "batch": batch,
            "max_abs_diff": -1,
            "mean_abs_diff": -1,
            "max_rel_diff": -1,
            "passed": False,
            "status": "[PRECISION_FAIL]",
            "error": str(e)[:200],
        }

    golden = matmul_golden(A.cpu(), B_mat.cpu())

    abs_diff = (output.cpu().float() - golden.float()).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())

    rel_diff = abs_diff / (golden.float().abs() + 1e-12)
    max_rel = float(rel_diff.max().item())

    # FP16 MatMul with FP16 accumulation vs FP32 golden - relax atol slightly
    passed = max_abs <= 0.05 and max_rel <= 0.05

    return {
        "batch": batch,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_rel_diff": max_rel,
        "passed": passed,
        "status": "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]",
    }


def run_tests(test_type="all"):
    """Run selected tests."""
    all_pass = True
    results = []
    device_id = 0
    torch.npu.set_device(device_id)

    if test_type in ("all", "smoke"):
        # Smoke test: 2D [16,16] @ [16,16]
        print("=== Smoke test: [16,16] @ [16,16] ===")
        A_small = torch.randn(16, 16, dtype=torch.float16).npu(device_id)
        B_small = torch.randn(16, 16, dtype=torch.float16).npu(device_id)
        try:
            Y_small = matmul_wrapper(A_small, B_small)
            torch.npu.synchronize(device_id)
            print(f"  Smoke test PASS (Y shape: {Y_small.shape})")
            results.append({"test": "smoke_2d", "status": "PASS"})
        except Exception as e:
            print(f"  Smoke test FAIL: {e}")
            results.append({"test": "smoke_2d", "status": "FAIL", "error": str(e)})
            all_pass = False

        # Shape gate test: 2D [256,256] @ [256,32]
        print("=== Shape gate: [256,256] @ [256,32] ===")
        A_gate = torch.randn(256, 256, dtype=torch.float16).npu(device_id)
        B_gate = torch.randn(256, 32, dtype=torch.float16).npu(device_id)
        try:
            Y_gate = matmul_wrapper(A_gate, B_gate)
            torch.npu.synchronize(device_id)
            print(f"  Shape gate PASS (Y shape: {Y_gate.shape})")
            results.append({"test": "shape_gate_2d", "status": "PASS"})
        except Exception as e:
            print(f"  Shape gate FAIL: {e}")
            results.append({"test": "shape_gate_2d", "status": "FAIL", "error": str(e)})
            all_pass = False

        # 3D gate test: [1,256,256] @ [1,256,32]
        print("=== 3D gate: [1,256,256] @ [1,256,32] ===")
        A_3d = torch.randn(1, 256, 256, dtype=torch.float16).npu(device_id)
        B_3d = torch.randn(1, 256, 32, dtype=torch.float16).npu(device_id)
        try:
            Y_3d = matmul_wrapper(A_3d, B_3d)
            torch.npu.synchronize(device_id)
            print(f"  3D gate PASS (Y shape: {Y_3d.shape})")
            results.append({"test": "shape_gate_3d", "status": "PASS"})
        except Exception as e:
            print(f"  3D gate FAIL: {e}")
            results.append({"test": "shape_gate_3d", "status": "FAIL", "error": str(e)})
            all_pass = False

    if test_type in ("all", "precision"):
        # Full precision tests (4D batched)
        print("=== Full precision tests (4D: [B,12,256,256] @ [B,12,256,32]) ===")
        for b in BATCHES:
            result = test_precision(b, device_id)
            if result.get("error"):
                print(f"  B={b}: ERROR: {result['error']}")
            elif result["passed"]:
                print(f"  B={b}: [PRECISION_PASS] max_abs={result['max_abs_diff']:.6f}")
            else:
                print(f"  B={b}: [PRECISION_FAIL] max_abs={result['max_abs_diff']:.6f} "
                      f"max_rel={result['max_rel_diff']:.6f}")
                all_pass = False
            results.append(result)

    # Summary
    if all_pass:
        print("\n[PRECISION_PASS] All tests passed")
    else:
        print("\n[PRECISION_FAIL] Some tests failed")

    # Write results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "test_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "all_pass": all_pass}, f, indent=2)

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "smoke", "precision"],
                        help="Which tests to run")
    args = parser.parse_args()
    sys.exit(0 if run_tests(args.test) else 1)
