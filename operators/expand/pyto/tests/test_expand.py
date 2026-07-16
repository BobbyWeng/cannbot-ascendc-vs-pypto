"""Test entry for Expand PyPTO with tri-state marking."""
import os, sys, json, torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from expand_golden import expand_golden
from expand_impl import expand_wrapper

BATCHES = [1, 2, 4, 8, 16, 32, 64]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

results = []
all_pass = True

for b in BATCHES:
    in_path = os.path.join(DATA_DIR, f"input_b{b}_fp16.bin")
    ref_path = os.path.join(DATA_DIR, f"reference_b{b}_fp16.bin")
    x = torch.from_numpy(np.fromfile(in_path, dtype=np.float16).reshape(b, 256, 1))
    ref = torch.from_numpy(np.fromfile(ref_path, dtype=np.float16).reshape(b, 256, 384))

    torch.npu.set_device(0)
    x_npu = x.npu(0)
    ref_npu = ref.npu(0)

    try:
        y = expand_wrapper(x_npu)
        diff = (y.cpu() != ref).sum().item()
        if diff == 0:
            status = "[PRECISION_PASS]"
        else:
            status = "[PRECISION_FAIL]"
            all_pass = False
    except Exception as e:
        status = f"[ERROR] {e}"
        all_pass = False

    print(f"B={b}: {status}")
    results.append({"batch": b, "status": status})

out = os.path.join(os.path.dirname(__file__), 'test_results.json')
with open(out, 'w') as f:
    json.dump({"results": results, "all_pass": all_pass}, f, indent=2)

if all_pass:
    print("[PRECISION_PASS] All batches passed")
else:
    print("[PRECISION_FAIL] Some batches failed")
