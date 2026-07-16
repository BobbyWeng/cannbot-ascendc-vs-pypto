"""Test entry for ReduceSum with three-state marking."""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from reduce_sum_golden import reduce_sum_golden
from reduce_sum_impl import reduce_sum_wrapper

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = (256, 384)
DTYPE = torch.float16
SEED = 20260715
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

COVERAGE_CASES = [
    "random_finite", "all_zero", "all_one", "pos_neg_cancel",
    "small_values", "large_values", "overflow_risk", "underflow_risk",
    "nan", "inf",
]


def test_reduce_sum():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    all_pass = True
    for B in BATCH_SIZES:
        for case in COVERAGE_CASES:
            shape = (B,) + SHAPE_TAIL
            x_path = os.path.join(DATA_DIR, f"x_b{B}_{case}.bin")
            x = torch.from_numpy(np.fromfile(x_path, dtype=torch.float16).reshape(shape))

            expected = reduce_sum_golden(x)

            x_npu = x.npu(device_id)
            y = reduce_sum_wrapper(x_npu)
            actual = y.cpu()

            abs_diff = (actual.float() - expected.float()).abs()
            max_abs = abs_diff.max().item()
            rel_diff = abs_diff / (expected.float().abs() + 1e-12)
            max_rel = rel_diff.max().item()

            if max_abs <= 0.01 and max_rel <= 0.01:
                print(f"[PRECISION_PASS] B={B} {case}: max_abs={max_abs:.6f} max_rel={max_rel:.6f}")
            else:
                print(f"[PRECISION_FAIL] B={B} {case}: max_abs={max_abs:.6f} max_rel={max_rel:.6f}")
                all_pass = False

    if all_pass:
        print("[PRECISION_PASS] All batch sizes and cases passed")
    else:
        print("[PRECISION_FAIL] Some batch sizes or cases failed")
        sys.exit(1)


if __name__ == "__main__":
    test_reduce_sum()
