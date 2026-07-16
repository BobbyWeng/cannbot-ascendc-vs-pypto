"""Test entry for Equal with three-state marking."""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from equal_golden import equal_golden
from equal_impl import equal_wrapper

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = (256, 384)
DTYPE = torch.float16
SEED = 20260715
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')


def test_equal():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    all_pass = True
    for B in BATCH_SIZES:
        shape = (B,) + SHAPE_TAIL
        x1_path = os.path.join(DATA_DIR, f"x1_b{B}_fp16.bin")
        x2_path = os.path.join(DATA_DIR, f"x2_b{B}_fp16.bin")

        x1 = torch.from_numpy(np.fromfile(x1_path, dtype=np.float16).reshape(shape))
        x2 = torch.from_numpy(np.fromfile(x2_path, dtype=np.float16).reshape(shape))

        expected = equal_golden(x1, x2)

        x1_npu = x1.npu(device_id)
        x2_npu = x2.npu(device_id)

        y = equal_wrapper(x1_npu, x2_npu)
        actual = y.cpu().bool()

        max_diff = (actual != expected).sum().item()
        bitwise_match = torch.equal(actual, expected)

        if bitwise_match:
            print(f"[PRECISION_PASS] B={B}: bitwise equal")
        else:
            print(f"[PRECISION_FAIL] B={B}: mismatches={max_diff}")
            all_pass = False

    if all_pass:
        print("[PRECISION_PASS] All batch sizes passed")
    else:
        print("[PRECISION_FAIL] Some batch sizes failed")
        sys.exit(1)


if __name__ == "__main__":
    test_equal()
