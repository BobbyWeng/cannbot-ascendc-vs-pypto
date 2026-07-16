"""PyPTO kernel implementation for Mul — receives 4D [B,3,4,256,32], processes via 2D JIT."""
import os
import torch
import pypto
import pypto.op


@pypto.frontend.jit
def mul_kernel_2d(x1: pypto.Tensor([], pypto.DT_FP16),
                  x2: pypto.Tensor([], pypto.DT_FP16),
                  y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(1024, 2048)
    y.move(pypto.op.mul(x1, x2))


_mul_cache = {}

def mul_wrapper(x1, x2):
    orig_shape = x1.shape
    key = orig_shape
    if key not in _mul_cache:
        x1_2d = x1.reshape(-1, orig_shape[-1])
        x2_2d = x2.reshape(-1, orig_shape[-1])
        y_2d = torch.empty(x1_2d.shape, dtype=torch.float16, device=x1.device)
        _mul_cache[key] = (x1_2d, x2_2d, y_2d)
    else:
        x1_view, x2_view, y_2d = _mul_cache[key]
        x1_2d = x1.reshape(-1, orig_shape[-1])
        x2_2d = x2.reshape(-1, orig_shape[-1])

    mul_kernel_2d(x1_2d, x2_2d, y_2d)
    return y_2d.reshape(orig_shape)
