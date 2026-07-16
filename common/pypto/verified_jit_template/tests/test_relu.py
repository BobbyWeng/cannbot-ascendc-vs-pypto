"""Minimal test for verified JIT template."""
import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'golden'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from relu_golden import relu_golden
from relu_impl import relu_wrapper


def source_introspection():
    import inspect
    print("=" * 60)
    print("SOURCE INTROSPECTION")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"cwd: {os.getcwd()}")
    print(f"sys.path[0]: {sys.path[0]}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")

    from relu_impl import relu_kernel_2d, relu_wrapper
    for name, fn in [("relu_kernel_2d", relu_kernel_2d), ("relu_wrapper", relu_wrapper)]:
        print(f"\n--- {name} ---")
        print(f"  __module__: {fn.__module__}")
        print(f"  __qualname__: {fn.__qualname__}")
        try:
            print(f"  inspect.getfile: {inspect.getfile(fn)}")
        except Exception as e:
            print(f"  inspect.getfile FAILED: {e}")
        try:
            print(f"  inspect.getsourcefile: {inspect.getsourcefile(fn)}")
        except Exception as e:
            print(f"  inspect.getsourcefile FAILED: {e}")
        try:
            src = inspect.getsource(fn)
            print(f"  inspect.getsource: OK ({len(src)} chars)")
            print(f"  First 100 chars: {src[:100].strip()}")
        except Exception as e:
            print(f"  inspect.getsource FAILED: {e}")


def test_relu():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    import torch_npu
    torch.npu.set_device(device_id)

    shape = (1, 12, 256, 32)
    x = torch.randn(shape, dtype=torch.float16)
    x_npu = x.npu()
    expected = relu_golden(x)

    y = relu_wrapper(x_npu)
    actual = y.cpu().to(torch.float16)

    max_diff = (actual - expected).abs().max().item()
    bitwise_match = torch.equal(actual, expected)

    if bitwise_match:
        print(f"[PRECISION_PASS] B=1: bitwise equal, max_diff={max_diff:.6f}")
    else:
        print(f"[PRECISION_FAIL] B=1: max_diff={max_diff:.6f}")
        sys.exit(1)


if __name__ == "__main__":
    source_introspection()
    test_relu()
    print("\n[PRECISION_PASS] Verified JIT template: SUCCESS")
