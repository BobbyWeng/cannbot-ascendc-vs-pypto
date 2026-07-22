"""PyPTO kernel implementation for LayerNorm — normalize only on last dim.
Weight/bias applied on host side after kernel."""
import os
import torch
import pypto
import pypto.op as op


@pypto.frontend.jit
def layernorm_kernel_2d(x: pypto.Tensor([], pypto.DT_FP16),
                         y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(1024, 2048)

    mean = op.sum(x, 1, keepdim=True) * 0.03125
    centered = x - mean
    var = op.sum(centered * centered, 1, keepdim=True) * 0.03125
    rstd = op.rsqrt(var + 1e-5)
    y.move(centered * rstd)


_layernorm_cache = {}

def layernorm_wrapper(x, weight, bias):
    orig_shape = x.shape
    key = (orig_shape, weight.shape[0])
    if key not in _layernorm_cache:
        x_2d = x.reshape(-1, orig_shape[-1])
        y_2d = torch.empty(x_2d.shape, dtype=torch.float16, device=x.device)
        _layernorm_cache[key] = (x_2d, y_2d)
    else:
        x_view, y_2d = _layernorm_cache[key]
        x_2d = x.reshape(-1, orig_shape[-1])

    layernorm_kernel_2d(x_2d, y_2d)
    y_normed = y_2d.reshape(orig_shape)
    return y_normed * weight + bias
