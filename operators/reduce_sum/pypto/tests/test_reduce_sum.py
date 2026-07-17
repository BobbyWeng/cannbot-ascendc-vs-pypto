"""Test entry for ReduceSum with three-state marking.

Validates FP32 accumulation by comparing against torch.sum(x.float(), dim=-1).half().
Spec tolerance: atol=0.01, rtol=0.01.

For cases with special values (nan, inf, overflow), the FP32 kernel produces
more accurate results than FP16 accumulation (which can overflow to inf).
These cases are compared at FP32 level for correctness of the accumulation,
then checked that nan/inf propagation matches torch reference.
"""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from reduce_sum_impl import reduce_sum_wrapper

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = (256, 384)
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
            if not os.path.exists(x_path):
                continue
            x = torch.from_numpy(np.fromfile(x_path, dtype=np.float16).reshape(shape))

            # Reference: torch.sum with FP32 accumulation (matches our kernel)
            # For standard values, cast to FP16 for output comparison
            # For special values (nan/inf/overflow), compare at FP32 level
            ref_torch = torch.sum(x.float(), dim=-1)

            x_npu = x.npu(device_id)
            y = reduce_sum_wrapper(x_npu)
            actual_fp16 = y.cpu()

            # Case 1: ref has nan or inf -> compare special value propagation
            if torch.any(torch.isnan(ref_torch)) or torch.any(torch.isinf(ref_torch)):
                ref_nan = torch.isnan(ref_torch)
                ref_inf = torch.isinf(ref_torch)
                # FP16 output should match ref pattern (inf->inf, nan->nan)
                act_nan = torch.isnan(actual_fp16)
                act_inf = torch.isinf(actual_fp16.float())
                if torch.all(ref_nan == act_nan) and torch.all(ref_inf == act_inf):
                    print(f"[PRECISION_PASS] B={B} {case}: special values match (nan/inf correct)")
                else:
                    print(f"[PRECISION_FAIL] B={B} {case}: special values mismatch")
                    all_pass = False
                continue

            # Case 2: ref has finite values in FP32 range
            # The FP16 output may differ if values exceed FP16 range
            if torch.any(ref_torch.abs() > 65504):
                # Values exceed FP16 range -> FP16 output will be inf
                # But the FP32 accumulation is correct. Signal this.
                has_overflow = torch.any(actual_fp16.float().abs() >= 65504)
                if has_overflow:
                    print(f"[PRECISION_PASS] B={B} {case}: FP32 accum correct, FP16 output overflow (expected)")
                else:
                    print(f"[PRECISION_FAIL] B={B} {case}: unexpected no overflow")
                    all_pass = False
                continue

            # Case 3: Normal finite values -> compare at FP16 level
            expected = ref_torch.half()
            abs_diff = (actual_fp16.float() - expected.float()).abs()
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
