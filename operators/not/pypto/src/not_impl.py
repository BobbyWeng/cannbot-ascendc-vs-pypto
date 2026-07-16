import torch
import pypto
import pypto.op


@pypto.frontend.jit
def not_kernel(x: pypto.Tensor([], pypto.DT_BOOL),
               y: pypto.Tensor([], pypto.DT_BOOL)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.op.logical_not(x))


def not_wrapper(x: torch.Tensor) -> torch.Tensor:
    """Host wrapper: returns logical_not(x) using 2D reshape."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    y_2d = torch.empty(x_2d.shape, dtype=torch.uint8, device=x.device)
    not_kernel(x_2d, y_2d)
    return y_2d.reshape(orig_shape)
