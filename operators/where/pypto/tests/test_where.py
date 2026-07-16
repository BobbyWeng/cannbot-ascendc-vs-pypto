"""Test entry for Where with three-state marking."""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from where_golden import where_golden
from where_impl import where_wrapper

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = (256, 384)
DTYPE = torch.float16
SEED = 20260715
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')


def test_where():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    all_pass = True
    for B in BATCH_SIZES:
        shape = (B,) + SHAPE_TAIL
        cond_path = os.path.join(DATA_DIR, f"condition_b{B}_bool.bin")
        x1_path = os.path.join(DATA_DIR, f"x1_b{B}_fp16.bin")
        x2_path = os.path.join(DATA_DIR, f"x2_b{B}_fp16.bin")

        condition = torch.from_numpy(np.fromfile(cond_path, dtype=np.uint8).reshape(shape))
        x1 = torch.from_numpy(np.fromfile(x1_path, dtype=np.float16).reshape(shape))
        x2 = torch.from_numpy(np.fromfile(x2_path, dtype=np.float16).reshape(shape))

        expected = where_golden(condition, x1, x2)

        cond_npu = condition.npu(device_id)
        x1_npu = x1.npu(device_id)
        x2_npu = x2.npu(device_id)

        y = where_wrapper(cond_npu, x1_npu, x2_npu)
        actual = y.cpu().to(torch.float16)

        max_diff = (actual - expected).abs().max().item()
        bitwise_match = torch.equal(actual, expected)

        if bitwise_match:
            print(f"[PRECISION_PASS] B={B}: bitwise equal")
        else:
            print(f"[PRECISION_FAIL] B={B}: max_diff={max_diff:.6f}")
            all_pass = False

    if all_pass:
        print("[PRECISION_PASS] All batch sizes passed")
    else:
        print("[PRECISION_FAIL] Some batch sizes failed")
        sys.exit(1)


if __name__ == "__main__":
    test_where()
