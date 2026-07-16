"""Test entry for Expand with three-state marking."""
import os
import sys
import torch
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from expand_golden import expand_golden
from expand_impl import expand_wrapper


def source_introspection(log_dir):
    import inspect
    from expand_impl import expand_row
    lines = []
    lines.append("=" * 60)
    lines.append("SOURCE INTROSPECTION")
    lines.append("=" * 60)
    lines.append(f"Python executable: {sys.executable}")
    lines.append(f"cwd: {os.getcwd()}")
    lines.append(f"sys.path: {sys.path}")

    orig = getattr(expand_row, '_original_func', expand_row)
    lines.append(f"function type: {type(expand_clone_kernel)}")
    lines.append(f"__module__: {getattr(expand_clone_kernel, '__module__', 'N/A')}")
    try:
        src = inspect.getsource(orig)
        lines.append(f"inspect.getsource: OK ({len(src)} chars)")
        lines.append(f"Source:\n{src}")
    except Exception as e:
        lines.append(f"inspect.getsource FAILED: {e}")
    try:
        f = inspect.getfile(orig)
        lines.append(f"inspect.getfile: {f}")
    except Exception as e:
        lines.append(f"inspect.getfile FAILED: {e}")
    try:
        sf = inspect.getsourcefile(orig)
        lines.append(f"inspect.getsourcefile: {sf}")
    except Exception as e:
        lines.append(f"inspect.getsourcefile FAILED: {e}")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "source_introspection.log")
    with open(log_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def test_expand():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    BATCHES = [1, 2, 4, 8, 16, 32, 64]
    all_pass = True
    results = []

    for B in BATCHES:
        x = torch.randn(B, 256, 1, dtype=torch.float16)
        expected = expand_golden(x)
        x_npu = x.npu()
        y = expand_wrapper(x_npu)
        actual = y.cpu()

        max_diff = (actual - expected).abs().max().item()
        bitwise_match = torch.equal(actual, expected)

        results.append({"batch": B, "shape": list(x.shape), "output_shape": list(y.shape), "bitwise": bool(bitwise_match), "max_diff": float(max_diff)})
        if bitwise_match:
            print(f"[PRECISION_PASS] B={B}: bitwise equal (shape {list(x.shape)} -> {list(y.shape)})")
        else:
            print(f"[PRECISION_FAIL] B={B}: max_diff={max_diff:.6f}")
            all_pass = False

    out = {"operator": "expand", "variant": "pypto", "results": results, "all_pass": all_pass}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'correctness_results.json')
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    if all_pass:
        print("[PRECISION_PASS] All batch sizes passed")
    else:
        print("[PRECISION_FAIL] Some batch sizes failed")
        sys.exit(1)


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    source_introspection(log_dir)
    test_expand()
