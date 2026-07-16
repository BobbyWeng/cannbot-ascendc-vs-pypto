import torch
import pypto
import pypto.op


@pypto.frontend.jit
def or_kernel(x1: pypto.Tensor([], pypto.DT_UINT8),
              x2: pypto.Tensor([], pypto.DT_UINT8),
              y: pypto.Tensor([], pypto.DT_UINT8)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.bitwise_or(x1, x2))


def or_wrapper(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Host wrapper for logical OR on BOOL (uint8) tensors."""
    orig_shape = x1.shape
    x1_2d = x1.reshape(-1, orig_shape[-1])
    x2_2d = x2.reshape(-1, orig_shape[-1])
    y_2d = torch.empty(x1_2d.shape, dtype=torch.uint8, device=x1.device)
    or_kernel(x1_2d, x2_2d, y_2d)
    return y_2d.reshape(orig_shape)
