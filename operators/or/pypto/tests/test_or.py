"""Test entry for Or (LogicalOr) with three-state marking."""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from or_golden import or_golden
from or_impl import or_wrapper

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = (256, 384)
DTYPE = torch.uint8
SEED = 20260715
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

INPUT_COMBINATIONS = [
    "false_false", "false_true", "true_false", "true_true",
    "random_mask", "sparse", "dense",
]


def test_or():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    all_pass = True
    for B in BATCH_SIZES:
        for variant in INPUT_COMBINATIONS:
            shape = (B,) + SHAPE_TAIL
            x1_path = os.path.join(DATA_DIR, f"x1_b{B}_{variant}.bin")
            x2_path = os.path.join(DATA_DIR, f"x2_b{B}_{variant}.bin")

            x1 = torch.from_numpy(np.fromfile(x1_path, dtype=np.uint8).reshape(shape))
            x2 = torch.from_numpy(np.fromfile(x2_path, dtype=np.uint8).reshape(shape))

            expected = or_golden(x1, x2)

            x1_npu = x1.npu(device_id)
            x2_npu = x2.npu(device_id)

            y = or_wrapper(x1_npu, x2_npu)
            actual = y.cpu().to(torch.uint8)

            bitwise_match = torch.equal(actual, expected)

            if bitwise_match:
                print(f"[PRECISION_PASS] B={B} {variant}: bitwise equal")
            else:
                max_diff = (actual != expected).sum().item()
                print(f"[PRECISION_FAIL] B={B} {variant}: mismatches={max_diff}")
                all_pass = False

    if all_pass:
        print("[PRECISION_PASS] All batch sizes and variants passed")
    else:
        print("[PRECISION_FAIL] Some batch sizes or variants failed")
        sys.exit(1)


if __name__ == "__main__":
    test_or()
