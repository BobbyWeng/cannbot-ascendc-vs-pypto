#!/usr/bin/env python3
import torch
import torch_npu
import pypto

torch.npu.set_device(0)

@pypto.frontend.jit
def eq_kernel_mini(x1: pypto.Tensor([], pypto.DT_FP16),
                   x2: pypto.Tensor([], pypto.DT_FP16),
                   y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(128, 32)
    y.move(pypto.eq(x1, x2))


def test_minimal():
    x1 = torch.randn(1, 32, dtype=torch.float16, device='npu:0')
    x2 = x1.clone()
    y = torch.zeros(1, 32, dtype=torch.float16, device='npu:0')
    eq_kernel_mini(x1, x2, y)
    torch.npu.synchronize()
    print(f'PASS: sum={y.sum().item()}')


def test_full():
    import numpy as np
    b = 1
    shape = [b, 256, 384]
    x1 = torch.from_numpy(np.fromfile(f'operators/equal/data/x1_b{b}_fp16.bin', dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(f'operators/equal/data/x2_b{b}_fp16.bin', dtype=np.float16).reshape(shape))
    y = torch.empty(shape, dtype=torch.float16, device='npu:0')
    x1_2d = x1.reshape(-1, shape[-1]).npu(0)
    x2_2d = x2.reshape(-1, shape[-1]).npu(0)
    y_2d = torch.empty(x1_2d.shape, dtype=torch.float16, device='npu:0')
    eq_kernel_mini(x1_2d, x2_2d, y_2d)
    torch.npu.synchronize()
    ref = torch.from_numpy(np.fromfile(f'operators/equal/data/reference_b{b}_bool.bin', dtype=np.uint8).reshape(shape)).bool()
    out = y_2d.reshape(shape).bool()
    mismatches = (out.cpu() != ref).sum().item()
    print(f'Full test: mismatches={mismatches}/{y.numel()}')


if __name__ == '__main__':
    try:
        test_minimal()
    except Exception as e:
        print(f'Minimal test FAILED: {e}')
    try:
        test_full()
    except Exception as e:
        print(f'Full test FAILED: {e}')
