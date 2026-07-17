"""PyPTO kernel implementation for Div — broadcast division with last-dim broadcast.

Kernel shapes: X1 [B,12,256,256], X2 [B,12,256,1], Y [B,12,256,256].
X2 broadcast along last dim is handled via 2D reshape ([-1, 256] / [-1, 1])
with tile shapes (128, 1024) to avoid CompileFunction backend failure.

Diagnostic findings (2026-07-17):
  - pypto.op.div works for 2D broadcast [256,256]/[256,1] with tile=(128,1024)
  - pypto.op.div FAILS at CompileFunction for:
    * tile_shape[0] >= 256 (even on same-shape 2D)
    * 3D/4D tensor shapes regardless of tile shape
    * FP32 dtype regardless of shape
  - reciprocal+mul has FP16 precision issue (5% elements exceed 1e-3, max_abs=0.0156)
  - Strategy: reshape 4D to 2D, use native div with tile=(128,1024), reshape back
"""
import os
import torch
import pypto
import pypto.op


@pypto.frontend.jit
def div_kernel_2d(x1: pypto.Tensor([], pypto.DT_FP16),
                  x2: pypto.Tensor([], pypto.DT_FP16),
                  y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.op.div(x1, x2))


def div_wrapper(x1, x2):
    """Broadcast division using 2D reshape workaround.
    
    Reshapes 4D tensors to 2D, applies native div, then reshapes back.
    Uses pypto.zeros() to initialize output tensor.
    """
    orig_shape = x1.shape
    orig_x2_shape = x2.shape
    x1_2d = x1.reshape(-1, orig_shape[-1])
    x2_2d = x2.reshape(-1, orig_x2_shape[-1])
    y_2d = torch.zeros(x1_2d.shape, dtype=x1.dtype, device=x1.device)
    div_kernel_2d(x1_2d, x2_2d, y_2d)
    return y_2d.reshape(orig_shape)
