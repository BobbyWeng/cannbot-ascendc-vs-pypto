"""
Case E: Four-input chained Add, all B = {1,2,4,8,16,32,64}
  Y = ((X1+X2)+X3)+X4  shape: [B,256,384] FP16
"""
import sys
import os
import traceback

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = (256, 384)
result = {"case": "E", "description": f"4-input chained add all batch sizes",
          "batch_sizes": BATCH_SIZES, "results": {}, "exit_code": -1, "status": "not_run"}

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, '..', 'src'))

import torch, numpy as np, warnings
warnings.filterwarnings("ignore")
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
import torch_npu
torch.npu.set_device(device_id)
from add_impl import add_4

torch.manual_seed(20260715)
all_pass = True

for B in BATCH_SIZES:
    shape = (B,) + SHAPE_TAIL
    x1 = torch.randn(*shape, dtype=torch.float16)
    x2 = torch.randn(*shape, dtype=torch.float16)
    x3 = torch.randn(*shape, dtype=torch.float16)
    x4 = torch.randn(*shape, dtype=torch.float16)
    expected = ((x1 + x2) + x3) + x4

    try:
        y = add_4(x1.npu(), x2.npu(), x3.npu(), x4.npu())
        torch.npu.synchronize(device_id)
        match = torch.equal(y.cpu(), expected)
        result["results"][str(B)] = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
            max_diff = (y.cpu() - expected).abs().max().item()
            print(f"[RESULT] B={B}: FAIL  max_diff={max_diff}")
        else:
            print(f"[RESULT] B={B}: PASS  bitwise_equal=True")
    except Exception as e:
        result["results"][str(B)] = f"ERROR: {type(e).__name__}"
        all_pass = False
        print(f"[RESULT] B={B}: ERROR — {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()

result["status"] = "ALL_PASS" if all_pass else "SOME_FAILED"
result["exit_code"] = 0 if all_pass else 1
print(f"[RESULT] Case E: {result['status']}")
