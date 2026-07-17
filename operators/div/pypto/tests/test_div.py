"""Test entry for Div with three-state marking — broadcast [B,12,256,256] / [B,12,256,1]."""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from div_golden import div_golden
from div_impl import div_wrapper

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
KERNEL_TAIL = (12, 256, 256)
X2_KERNEL_TAIL = (12, 256, 1)
DTYPE = torch.float16
SEED = 20260715
ATOL = 1e-3
RTOL = 1e-3


def test_div():
    torch.manual_seed(SEED)
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    all_pass = True
    for B in BATCH_SIZES:
        shape = (B,) + KERNEL_TAIL
        x2_shape = (B,) + X2_KERNEL_TAIL
        x1 = torch.randn(shape, dtype=DTYPE) * 2.0
        x2_sign = torch.sign(torch.randn(x2_shape, dtype=DTYPE))
        x2_mag = torch.rand(x2_shape, dtype=DTYPE) * 3.75 + 0.25
        x2 = x2_sign * x2_mag
        x2 = x2.clamp(-4.0, -0.25) + x2.clamp(0.25, 4.0)
        x2_mask = (x2.abs() >= 0.25)
        x2 = torch.where(x2_mask, x2, torch.ones_like(x2) * 1.0)

        x1_npu = x1.npu()
        x2_npu = x2.npu()
        expected = div_golden(x1, x2)

        y = div_wrapper(x1_npu, x2_npu)
        actual = y.cpu().to(torch.float16)

        max_abs = (actual - expected).abs().max().item()
        max_rel = ((actual - expected).abs() / (expected.abs() + 1e-12)).max().item()

        try:
            np.testing.assert_allclose(actual.numpy(), expected.numpy(),
                                       rtol=RTOL, atol=ATOL)
            print(f"[PRECISION_PASS] B={B}: atol={ATOL} rtol={RTOL} max_abs={max_abs:.6e} max_rel={max_rel:.6e}")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] B={B}: max_abs={max_abs:.6e} max_rel={max_rel:.6e}")
            print(f"  {e}", file=sys.stderr)
            all_pass = False

    if all_pass:
        print("[PRECISION_PASS] All batch sizes passed")
    else:
        print("[PRECISION_FAIL] Some batch sizes failed")
        sys.exit(1)


if __name__ == "__main__":
    test_div()
