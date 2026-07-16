"""Test entry for Transpose with three-state marking."""
import os
import sys
import torch
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from transpose_golden import transpose_golden
from transpose_impl import transpose_wrapper


def test_transpose():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    # Test small cases first: [B, H, W] -> [B, W, H]
    cases = [
        (1, 16, 32),
        (1, 32, 16),
        (1, 256, 384),
    ]
    all_pass = True

    for B, H, W in cases:
        x = torch.randn(B, H, W, dtype=torch.float16)
        expected = transpose_golden(x)
        x_npu = x.npu()
        try:
            y = transpose_wrapper(x_npu)
            actual = y.cpu()
            max_diff = (actual - expected).abs().max().item()
            bitwise_match = torch.equal(actual, expected)
            if bitwise_match:
                print(f"[PRECISION_PASS] [{B},{H},{W}] -> [{B},{W},{H}]: bitwise equal")
            else:
                print(f"[PRECISION_FAIL] [{B},{H},{W}] -> [{B},{W},{H}]: max_diff={max_diff:.6f}")
                all_pass = False
        except Exception as e:
            print(f"[PRECISION_FAIL] [{B},{H},{W}]: {e}")
            all_pass = False

    if all_pass:
        print("[PRECISION_PASS] All shapes passed")
    else:
        print("[PRECISION_FAIL] Some shapes failed")
        sys.exit(1)


if __name__ == "__main__":
    test_transpose()
