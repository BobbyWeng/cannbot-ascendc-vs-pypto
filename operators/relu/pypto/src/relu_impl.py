"""PyPTO kernel implementation for ReLU — per-batch JIT with 1-row tensors.

Workaround for d1c290f36 CompileFunction regression:
- JIT kernel only stable with single-row 2D tensors (shape [1, N])
- Multi-row tensors trigger host_machine.cpp:179 CompileFunction crash
- Strategy: per-batch-element loop, each call shape [1, inner_size]
- Historical: flatten-to-2D approach worked under earlier PyPTO versions
"""
import os
import torch
import pypto
import pypto.op


@pypto.frontend.jit
def relu_kernel_2d(x: pypto.Tensor([], pypto.DT_FP16),
                   y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(1024, 2048)
    y.move(pypto.op.relu(x))


def relu_wrapper(x):
    orig_shape = x.shape
    batch_size = orig_shape[0]
    inner_size = orig_shape[1:].numel()
    y = torch.empty(orig_shape, dtype=torch.float16, device=x.device)
    for b in range(batch_size):
        xb = x[b].ravel().reshape(1, inner_size)
        yb = y[b].ravel().reshape(1, inner_size)
        relu_kernel_2d(xb, yb)
    return y
