"""
Case A: FP16 [1,32] + [1,32] — minimal binary add
Purpose: Test if pypto.op.add works at all in the simplest possible case
"""
import sys
import os
import traceback

# --- Configuration ---
SHAPE = (1, 32)
DTYPE = "float16"

# Results dict (to be populated)
result = {
    "case": "A",
    "description": "FP16 [1,32] + [1,32] binary add",
    "shape": str(SHAPE),
    "source_file": "",
    "exit_code": -1,
    "stdout": "",
    "stderr": "",
    "exception": None,
    "generated_ir": None,
    "lowering_log": None,
    "failing_pass": None,
    "correctness": None,
    "status": "not_run",
}

# --- Import the kernel module ---
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(test_dir, '..', 'src')
sys.path.insert(0, src_dir)
result["source_file"] = os.path.join(src_dir, "add_impl.py")

import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
import torch_npu
torch.npu.set_device(device_id)

# Clean import of our module (JIT functions must be in imported module)
from add_impl import add_binary_kernel

# --- Setup golden ---
torch.manual_seed(20260715)
x1 = torch.randn(*SHAPE, dtype=torch.float16)
x2 = torch.randn(*SHAPE, dtype=torch.float16)
expected = x1 + x2  # torch reference

x1_npu = x1.npu()
x2_npu = x2.npu()
y_npu = torch.empty(*SHAPE, dtype=torch.float16, device=f"npu:{device_id}")

# --- Execute ---
try:
    add_binary_kernel(x1_npu, x2_npu, y_npu)
    torch.npu.synchronize(device_id)
    actual = y_npu.cpu()
    match = torch.equal(actual, expected)
    result["correctness"] = "PASS" if match else "FAIL"
    result["exit_code"] = 0
    result["status"] = "PASS" if match else "FAIL (correctness)"
    print(f"[RESULT] Case A: {result['status']}")
    print(f"[RESULT] bitwise_equal={match}")
    if not match:
        max_diff = (actual - expected).abs().max().item()
        print(f"[RESULT] max_diff={max_diff}")
except Exception as e:
    result["status"] = "FAIL"
    result["exit_code"] = 1
    result["exception"] = {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }
    print(f"[RESULT] Case A: FAIL — {type(e).__name__}: {str(e)[:200]}")
    print(f"[TRACEBACK]\n{traceback.format_exc()}")
    result["stderr"] = str(e)
