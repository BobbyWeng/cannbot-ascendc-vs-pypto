"""Test entry for Mul with three-state marking — 4D [B,3,4,256,32]."""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mul_golden import mul_golden
from mul_impl import mul_wrapper

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = (3, 4, 256, 32)
DTYPE = torch.float16
SEED = 20260715

def test_mul():
    torch.manual_seed(SEED)
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    all_pass = True
    for B in BATCH_SIZES:
        shape = (B,) + SHAPE_TAIL
        x1 = torch.randn(shape, dtype=DTYPE)
        x2 = torch.randn(shape, dtype=DTYPE)
        x1_npu = x1.npu()
        x2_npu = x2.npu()
        expected = mul_golden(x1, x2)

        y = mul_wrapper(x1_npu, x2_npu)
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
    test_mul()
