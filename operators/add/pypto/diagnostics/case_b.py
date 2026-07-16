"""
Case B: FP16 [1,256,384] + [1,256,384] — experiment shape
"""
import sys
import os
import traceback

SHAPE = (1, 256, 384)
result = {"case": "B", "description": "FP16 [1,256,384] + [1,256,384] binary add",
          "shape": str(SHAPE), "exit_code": -1, "status": "not_run"}

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, '..', 'src'))

import torch, numpy as np, warnings
warnings.filterwarnings("ignore")
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
import torch_npu
torch.npu.set_device(device_id)
from add_impl import add_binary

torch.manual_seed(20260715)
x1 = torch.randn(*SHAPE, dtype=torch.float16)
x2 = torch.randn(*SHAPE, dtype=torch.float16)
expected = x1 + x2

try:
    y = add_binary(x1.npu(), x2.npu())
    torch.npu.synchronize(device_id)
    match = torch.equal(y.cpu(), expected)
    result["status"] = "PASS" if match else "FAIL"
    result["exit_code"] = 0
    print(f"[RESULT] Case B: {result['status']}  bitwise_equal={match}")
    if not match:
        print(f"[RESULT] max_diff={(y.cpu()-expected).abs().max().item()}")
except Exception as e:
    result["status"] = "FAIL"
    result["exit_code"] = 1
    result["exception"] = str(e)[:500]
    print(f"[RESULT] Case B: FAIL — {type(e).__name__}: {str(e)[:300]}")
    traceback.print_exc()
