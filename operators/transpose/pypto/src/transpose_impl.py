"""PyPTO kernel implementation for Transpose — 2D transpose via JIT."""
import torch
import pypto
import pypto.op


@pypto.frontend.jit
def transpose_2d(x: pypto.Tensor([], pypto.DT_FP16),
                 y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(64, 256)
    y.move(pypto.op.transpose(x, 0, 1))


def transpose_wrapper(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    B, H, W = orig_shape[0], orig_shape[1], orig_shape[2]
    # For each batch, transpose [H,W] -> [W,H]
    y = torch.empty(B, W, H, dtype=torch.float16, device=x.device)
    x_2d = x.reshape(-1, W)
    for b in range(B):
        xs = x_2d[b*H:(b+1)*H]  # [H, W]
        ys = y[b]                # [W, H]
        transpose_2d(xs, ys)
    return y
