"""Test entry for LayerNorm with three-state marking — 4D [B,256,32]."""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from layernorm_golden import layernorm_golden
from layernorm_impl import layernorm_wrapper

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
NORM_SHAPE = (256, 32)
DTYPE = torch.float16
SEED = 20260715

def test_layernorm():
    torch.manual_seed(SEED)
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    rng = torch.Generator()
    rng.manual_seed(SEED + 999)
    weight = torch.randn(NORM_SHAPE, dtype=DTYPE, generator=rng)
    bias = torch.randn(NORM_SHAPE, dtype=DTYPE, generator=rng)
    weight_npu = weight.npu()
    bias_npu = bias.npu()
    eps = 1e-5

    all_pass = True
    for B in BATCH_SIZES:
        shape = (B,) + NORM_SHAPE
        x = torch.randn(shape, dtype=DTYPE)
        x_npu = x.npu()
        expected = layernorm_golden(x, weight, bias, eps)

        y = layernorm_wrapper(x_npu, weight_npu, bias_npu)
        actual = y.cpu().to(torch.float16)

        abs_diff = (actual.float() - expected.float()).abs()
        rel_diff = abs_diff / (expected.float().abs() + 1e-12)
        max_abs = abs_diff.max().item()
        max_rel = rel_diff.max().item()
        mismatch = int((abs_diff > 0.01).sum().item()) if max_abs > 0.01 else 0
        mismatch_rel = int((rel_diff > 0.001).sum().item()) if max_rel > 0.001 else 0

        passed = (mismatch == 0) or (mismatch_rel == 0)

        if passed:
            print(f"[PRECISION_PASS] B={B}: max_abs={max_abs:.6f}, max_rel={max_rel:.6f}")
        else:
            print(f"[PRECISION_FAIL] B={B}: max_abs={max_abs:.6f}, max_rel={max_rel:.6f}, mismatches={mismatch}")
            all_pass = False

    if all_pass:
        print("[PRECISION_PASS] All batch sizes passed")
    else:
        print("[PRECISION_FAIL] Some batch sizes failed")
        sys.exit(1)

if __name__ == "__main__":
    test_layernorm()
