#!/usr/bin/env python3
import torch
import torch_npu
import pypto
import numpy as np

torch.npu.set_device(0)


@pypto.frontend.jit
def eq_kernel_mini(x1, x2, y):
    pypto.set_vec_tile_shapes(128, 32)
    y.move(pypto.eq(x1, x2))


def test_equal_full():
    b = 1
    shape = [b, 256, 384]
    DATA_DIR = "/mnt/workspace/cannbot_ascendc_vs_pypto/operators/equal/data"
    x1 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x1_b{b}_fp16.bin", dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/x2_b{b}_fp16.bin", dtype=np.float16).reshape(shape))
    x1_npu = x1.npu(0)
    x2_npu = x2.npu(0)

    # Test 1: [1,32] minimal
    print("=== Test 1: [1,32] FP16 ===")
    x1_small = x1_npu[:, :, :32].contiguous()
    x2_small = x2_npu[:, :, :32].contiguous()
    y_small = torch.empty(1, 256, 32, dtype=torch.float16, device='npu:0')
    x1_2d = x1_small.reshape(-1, 32)
    x2_2d = x2_small.reshape(-1, 32)
    y_2d = y_small.reshape(-1, 32)
    eq_kernel_mini(x1_2d, x2_2d, y_2d)
    torch.npu.synchronize()
    y_cpu = y_2d.cpu().bool()
    ref = torch.from_numpy(np.fromfile(f"{DATA_DIR}/reference_b{b}_bool.bin", dtype=np.uint8).reshape(shape)[:, :, :32]).bool()
    mismatches = (y_cpu != ref).sum().item()
    print(f"  Mismatches: {mismatches}/{y_cpu.numel()}")
    print(f"  Output unique: {torch.unique(y_cpu).tolist()}")
    print(f"  Output sum: {y_2d.float().sum().item()}")

    # Test 2: output dtype = uint8 (BOOL)
    print("\n=== Test 2: [1,32] FP16 -> BOOL ===")

    @pypto.frontend.jit
    def eq_bool_kernel(x1, x2, y):
        pypto.set_vec_tile_shapes(128, 32)
        y.move(pyto.eq(x1, x2))

    y_bool = torch.empty(1, 256, 32, dtype=torch.uint8, device='npu:0')
    try:
        eq_bool_kernel(x1_2d, x2_2d, y_bool)
        torch.npu.synchronize()
        print(f"  BOOL dtype works, sum={y_bool.sum().item()}")
    except Exception as e:
        print(f"  BOOL dtype FAILED: {e}")

    # Test 3: [1,128]
    print("\n=== Test 3: [1,128] FP16 ===")
    x1_128 = x1_npu[:, :, :128].contiguous()
    x2_128 = x2_npu[:, :, :128].contiguous()
    y_128 = torch.empty(1, 256, 128, dtype=torch.float16, device='npu:0')
    x1_2d = x1_128.reshape(-1, 128)
    x2_2d = x2_128.reshape(-1, 128)
    y_2d = y_128.reshape(-1, 128)
    eq_kernel_mini(x1_2d, x2_2d, y_2d)
    torch.npu.synchronize()
    ref128 = torch.from_numpy(np.fromfile(f"{DATA_DIR}/reference_b{b}_bool.bin", dtype=np.uint8).reshape(shape)[:, :, :128]).bool()
    y_cpu = y_2d.cpu().bool()
    mismatches = (y_cpu != ref128).sum().item()
    print(f"  Mismatches: {mismatches}/{y_cpu.numel()}")

    # Test 4: full [1,256,384]
    print(f"\n=== Test 4: [1,256,384] FP16 ===")
    try:
        x1_2d = x1_npu.reshape(-1, 384)
        x2_2d = x2_npu.reshape(-1, 384)

        @pypto.frontend.jit
        def eq_full_kernel(x1, x2, y):
            pypto.set_vec_tile_shapes(128, 384)
            y.move(pypto.eq(x1, x2))

        y_full = torch.empty(256, 384, dtype=torch.float16, device='npu:0')
        eq_full_kernel(x1_2d, x2_2d, y_full)
        torch.npu.synchronize()
        y_cpu = y_full.reshape(shape).cpu().bool()
        ref = torch.from_numpy(np.fromfile(f"{DATA_DIR}/reference_b{b}_bool.bin", dtype=np.uint8).reshape(shape)).bool()
        mismatches = (y_cpu != ref).sum().item()
        print(f"  Full test: mismatches={mismatches}/{y_cpu.numel()}")
    except Exception as e:
        print(f"  Full test FAILED: {e}")


if __name__ == "__main__":
    test_equal_full()
