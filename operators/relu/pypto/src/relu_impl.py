"""PyPTO kernel implementation for ReLU — receives 4D, processes via 2D JIT."""
import os
import torch
import pypto
import pypto.op


@pypto.frontend.jit
def relu_kernel_2d(x: pypto.Tensor([], pypto.DT_FP16),
                   y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(1024, 2048)
    y.move(pypto.op.relu(x))


_relu_cache = {}

def relu_wrapper(x):
    orig_shape = x.shape
    key = orig_shape
    if key not in _relu_cache:
        x_2d = x.reshape(-1, orig_shape[-1])
        y_2d = torch.empty(x_2d.shape, dtype=torch.float16, device=x.device)
        _relu_cache[key] = (x_2d, y_2d)
    else:
        x_view, y_2d = _relu_cache[key]
        x_2d = x.reshape(-1, orig_shape[-1])

    relu_kernel_2d(x_2d, y_2d)
    return y_2d.reshape(orig_shape)
