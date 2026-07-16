#!/usr/bin/env python3
"""Minimal PyPTO Where diagnostic test."""
import os
import sys
import torch
import torch_npu
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from where_golden import where_golden
from where_impl import where_wrapper

torch.npu.set_device(0)
DATA_DIR = "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/where/data"

for B in [1, 2]:
    shape = [B, 256, 384]
    condition = torch.from_numpy(np.fromfile(f"{DATA_DIR}/condition_b{B}_bool.bin", dtype=np.uint8).reshape(shape))
    x1 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x1_b{B}_fp16.bin", dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x2_b{B}_fp16.bin", dtype=np.float16).reshape(shape))
    cond_npu = condition.npu(0)
    x1_npu = x1.npu(0)
    x2_npu = x2.npu(0)

    try:
        y = where_wrapper(cond_npu, x1_npu, x2_npu)
        torch.npu.synchronize()
        expected = where_golden(condition, x1, x2)
        actual = y.cpu().to(torch.float16)
        mismatches = torch.sum(actual.view(torch.uint16) != expected.view(torch.uint16)).item()
        print(f"[PYPTO_WHERE] B={B}: mismatches={mismatches}/{y.numel()}")
        if mismatches > 0:
            diff_mask = (actual.view(torch.uint16) != expected.view(torch.uint16))
            idx = torch.where(diff_mask)[0][:5]
            for i in idx:
                print(f"  pos {i}: actual={actual.flatten()[i].item():.8f}, expected={expected.flatten()[i].item():.8f}")
    except Exception as e:
        print(f"[PYPTO_WHERE] B={B}: FAILED: {e}")

# Test 2: same shape condition
print("\n=== Test: Condition same shape as X1/X2 ===")
B = 1
cond = torch.randint(0, 2, (256, 384), dtype=torch.uint8)
x1_t = torch.randn(256, 384, dtype=torch.float16)
x2_t = torch.randn(256, 384, dtype=torch.float16)
cond_npu = cond.npu(0)
x1_npu = x1_t.npu(0)
x2_npu = x2_t.npu(0)
try:
    y = where_wrapper(cond_npu, x1_npu, x2_npu)
    torch.npu.synchronize()
    expected = where_golden(cond, x1_t, x2_t)
    actual = y.cpu().to(torch.float16)
    mismatches = torch.sum(actual.view(torch.uint16) != expected.view(torch.uint16)).item()
    print(f"Same shape test: mismatches={mismatches}/{y.numel()}")
except Exception as e:
    print(f"Same shape test FAILED: {e}")
