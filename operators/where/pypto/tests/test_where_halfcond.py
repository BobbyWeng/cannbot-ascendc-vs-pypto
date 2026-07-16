#!/usr/bin/env python3
"""Test PyPTO Where with half condition (not uint8)."""
import torch
import torch_npu
import pypto
import pypto.op
import numpy as np

torch.npu.set_device(0)
DATA_DIR = "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/where/data"


@pypto.frontend.jit
def where_half_kernel(condition, x1, x2, y):
    pypto.set_vec_tile_shapes(64, 256)
    y.move(pypto.where(condition, x1, x2))


def where_half_wrapper(condition, x1, x2):
    orig_shape = x1.shape
    cond_2d = condition.reshape(-1, orig_shape[-1])
    x1_2d = x1.reshape(-1, orig_shape[-1])
    x2_2d = x2.reshape(-1, orig_shape[-1])
    y_2d = torch.empty(x1_2d.shape, dtype=torch.float16, device=x1.device)
    where_half_kernel(cond_2d, x1_2d, x2_2d, y_2d)
    return y_2d.reshape(orig_shape)


for B in [1, 2]:
    shape = [B, 256, 384]
    cond_path = f"{DATA_DIR}/condition_b{B}_bool.bin"
    condition = torch.from_numpy(np.fromfile(cond_path, dtype=np.uint8).reshape(shape))
    x1 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x1_b{B}_fp16.bin", dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x2_b{B}_fp16.bin", dtype=np.float16).reshape(shape))

    # Convert condition from uint8 to half (0→0.0, 1→1.0)
    condition_half = condition.to(torch.float16)

    cond_npu = condition_half.npu(0)
    x1_npu = x1.npu(0)
    x2_npu = x2.npu(0)

    try:
        y = where_half_wrapper(cond_npu, x1_npu, x2_npu)
        torch.npu.synchronize()
        # Reference: torch.where on float16 condition
        ref = torch.where(condition_half.bool(), x1, x2)
        actual = y.cpu().to(torch.float16)
        mismatches = torch.sum(actual.view(torch.uint16) != ref.view(torch.uint16)).item()
        print(f"[HALF_COND] B={B}: mismatches={mismatches}/{y.numel()}")
        if mismatches > 0:
            diff = (actual.view(torch.uint16) != ref.view(torch.uint16))
            idx = torch.where(diff)[0][:5]
            for i in idx:
                print(f"  [{i}]: act={actual.flatten()[i].item():.8f} ref={ref.flatten()[i].item():.8f}")
    except Exception as e:
        print(f"[HALF_COND] B={B}: FAILED: {e}")
