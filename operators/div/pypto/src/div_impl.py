"""PyPTO kernel implementation for Div — broadcast division with last-dim broadcast.

Kernel shapes: X1 [B,12,256,256], X2 [B,12,256,1], Y [B,12,256,256].
X2 broadcast along last dim is handled by the PyPTO runtime or explicit broadcast.
"""
import os
import torch
import pypto
import pypto.op

# Strategy: pypto.op.div if natively supported, else reciprocal_mul
# See API_REPORT.md for full audit


@pypto.frontend.jit
def div_kernel_2d(x1: pypto.Tensor([], pypto.DT_FP16),
                  x2: pypto.Tensor([], pypto.DT_FP16),
                  y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(1024, 2048)
    y.move(pypto.op.div(x1, x2))


_div_cache = {}

def div_wrapper(x1, x2):
    orig_shape = x1.shape
    orig_x2_shape = x2.shape
    dtype = x1.dtype
    device = x1.device
    key = (orig_shape, orig_x2_shape, dtype, device)
    if key not in _div_cache:
        x1_2d = x1.reshape(-1, orig_shape[-1])
        x2_2d = x2.reshape(-1, orig_x2_shape[-1])
        y_2d = torch.empty(x1_2d.shape, dtype=dtype, device=device)
        _div_cache[key] = (x1_2d, x2_2d, y_2d)
    else:
        x1_2d, x2_2d, y_2d = _div_cache[key]
        x1_2d = x1.reshape(-1, orig_shape[-1])
        x2_2d = x2.reshape(-1, orig_x2_shape[-1])

    div_kernel_2d(x1_2d, x2_2d, y_2d)
    return y_2d.reshape(orig_shape)
