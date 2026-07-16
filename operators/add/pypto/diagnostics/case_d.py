"""
Case D: Four-input chained Add via three binary Add calls
  t1 = add(x1, x2); t2 = add(t1, x3); y = add(t2, x4)
Shape: [1,256,384] FP16
"""
import sys
import os
import traceback

SHAPE = (1, 256, 384)
result = {"case": "D", "description": "4-input chained add [1,256,384] FP16",
          "shape": str(SHAPE), "exit_code": -1, "status": "not_run"}

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, '..', 'src'))

import torch, numpy as np, warnings
warnings.filterwarnings("ignore")
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
import torch_npu
torch.npu.set_device(device_id)
from add_impl import add_4

torch.manual_seed(20260715)
x1 = torch.randn(*SHAPE, dtype=torch.float16)
x2 = torch.randn(*SHAPE, dtype=torch.float16)
x3 = torch.randn(*SHAPE, dtype=torch.float16)
x4 = torch.randn(*SHAPE, dtype=torch.float16)
expected = ((x1 + x2) + x3) + x4

try:
    y = add_4(x1.npu(), x2.npu(), x3.npu(), x4.npu())
    torch.npu.synchronize(device_id)
    match = torch.equal(y.cpu(), expected)
    result["status"] = "PASS" if match else "FAIL"
    result["exit_code"] = 0
    print(f"[RESULT] Case D: {result['status']}  bitwise_equal={match}")
    if not match:
        print(f"[RESULT] max_diff={(y.cpu()-expected).abs().max().item()}")
except Exception as e:
    result["status"] = "FAIL"
    result["exit_code"] = 1
    result["exception"] = str(e)[:500]
    print(f"[RESULT] Case D: FAIL — {type(e).__name__}: {str(e)[:300]}")
    traceback.print_exc()
