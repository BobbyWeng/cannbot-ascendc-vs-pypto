#!/usr/bin/env python3
"""Test PyPTO Where with half condition via existing wrapper."""
import sys
sys.path.insert(0, "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/where/pypto/src")
sys.path.insert(0, "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/where/pypto/golden")
import torch
import torch_npu
import numpy as np
from where_impl import where_wrapper
from where_golden import where_golden

torch.npu.set_device(0)
DATA_DIR = "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/where/data"

for B in [1]:
    shape = [B, 256, 384]
    condition = torch.from_numpy(np.fromfile(f"{DATA_DIR}/condition_b{B}_bool.bin", dtype=np.uint8).reshape(shape))
    x1 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x1_b{B}_fp16.bin", dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x2_b{B}_fp16.bin", dtype=np.float16).reshape(shape))

    # Original test (uint8 condition)
    cond_npu = condition.npu(0)
    x1_npu = x1.npu(0)
    x2_npu = x2.npu(0)
    try:
        y = where_wrapper(cond_npu, x1_npu, x2_npu)
        torch.npu.synchronize()
        expected = where_golden(condition, x1, x2)
        actual = y.cpu().to(torch.float16)
        mismatches = torch.sum(actual.view(torch.uint16) != expected.view(torch.uint16)).item()
        print(f"[UINT8_COND] B={B}: mismatches={mismatches}/{y.numel()}")
    except Exception as e:
        print(f"[UINT8_COND] B={B}: FAILED: {str(e)[:200]}")

    # Same shape test with half condition
    print(f"\n=== Test half condition [1,256,384] ===")
    condition_half = condition.to(torch.float16)
    cond_half_npu = condition_half.npu(0)
    try:
        y = where_wrapper(cond_half_npu, x1_npu, x2_npu)
        torch.npu.synchronize()
        ref = torch.where(condition_half.bool(), x1, x2)
        actual = y.cpu().to(torch.float16)
        mismatches = torch.sum(actual.view(torch.uint16) != ref.view(torch.uint16)).item()
        print(f"[HALF_COND] B={B}: mismatches={mismatches}/{y.numel()}")
    except Exception as e:
        print(f"[HALF_COND] B={B}: FAILED: {str(e)[:200]}")
