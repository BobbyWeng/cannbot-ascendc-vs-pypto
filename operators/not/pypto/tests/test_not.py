"""Test entry for Not (LogicalNot) with three-state marking."""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from not_golden import not_golden
from not_impl import not_wrapper

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = (256, 384)
DTYPE = torch.uint8
SEED = 20260715
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

BOUNDARY_CASES = ["all_true", "all_false", "alternating", "random_mask", "sparse_true", "dense_true"]


def test_not():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    all_pass = True
    for B in BATCH_SIZES:
        for case in BOUNDARY_CASES:
            shape = (B,) + SHAPE_TAIL
            x_path = os.path.join(DATA_DIR, f"x_b{B}_{case}.bin")
            x = torch.from_numpy(np.fromfile(x_path, dtype=np.uint8).reshape(shape))

            expected = not_golden(x)

            x_npu = x.npu(device_id)
            y = not_wrapper(x_npu)
            actual = y.cpu()

            max_diff = (actual.float() - expected.float()).abs().max().item()
            bitwise_match = torch.equal(actual, expected)

            if bitwise_match:
                print(f"[PRECISION_PASS] B={B} {case}: bitwise equal")
            else:
                print(f"[PRECISION_FAIL] B={B} {case}: max_diff={max_diff:.6f}")
                all_pass = False

    if all_pass:
        print("[PRECISION_PASS] All batch sizes and cases passed")
    else:
        print("[PRECISION_FAIL] Some batch sizes or cases failed")
        sys.exit(1)


if __name__ == "__main__":
    test_not()
