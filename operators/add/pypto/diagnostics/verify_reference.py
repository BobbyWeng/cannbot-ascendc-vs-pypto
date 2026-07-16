"""
Verify PyPTO chained add against pre-generated reference data
"""
import sys
sys.path.insert(0, '/mnt/workspace/cannbot_ascendc_vs_pypto/operators/add/pypto/src')
import torch, torch_npu, warnings, os
import numpy as np
warnings.filterwarnings("ignore")
torch.npu.set_device(0)
from add_impl import add_4

DATA_DIR = '/mnt/workspace/cannbot_ascendc_vs_pypto/operators/add/data'
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = (256, 384)
SEED = 20260715

torch.manual_seed(SEED)
all_pass = True

for B in BATCH_SIZES:
    shape = (B,) + SHAPE_TAIL
    x1 = torch.from_numpy(np.fromfile(f'{DATA_DIR}/x1_b{B}_fp16.bin', dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(f'{DATA_DIR}/x2_b{B}_fp16.bin', dtype=np.float16).reshape(shape))
    x3 = torch.from_numpy(np.fromfile(f'{DATA_DIR}/x3_b{B}_fp16.bin', dtype=np.float16).reshape(shape))
    x4 = torch.from_numpy(np.fromfile(f'{DATA_DIR}/x4_b{B}_fp16.bin', dtype=np.float16).reshape(shape))
    ref = torch.from_numpy(np.fromfile(f'{DATA_DIR}/reference_b{B}_fp16.bin', dtype=np.float16).reshape(shape))

    y = add_4(x1.npu(), x2.npu(), x3.npu(), x4.npu())
    torch.npu.synchronize(0)
    match = torch.equal(y.cpu(), ref)
    print(f'B={B:2d}: {"PASS" if match else "FAIL"}  bitwise_equal={match}')
    if not match:
        max_diff = (y.cpu() - ref).abs().max().item()
        print(f'       max_diff={max_diff}')
        all_pass = False

print(f'Overall: {"ALL PASS" if all_pass else "SOME FAILED"}')
