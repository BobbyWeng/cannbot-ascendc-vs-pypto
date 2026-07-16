import torch
import pypto
import pypto.op


@pypto.frontend.jit
def where_kernel(condition: pypto.Tensor([], pypto.DT_UINT8),
                 x1: pypto.Tensor([], pypto.DT_FP16),
                 x2: pypto.Tensor([], pypto.DT_FP16),
                 y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(64, 256)
    y.move(pypto.where(condition, x1, x2))


def where_wrapper(condition: torch.Tensor,
                  x1: torch.Tensor,
                  x2: torch.Tensor) -> torch.Tensor:
    """Host wrapper: returns torch.where(condition, x1, x2) using 2D reshape."""
    orig_shape = x1.shape
    cond_2d = condition.reshape(-1, orig_shape[-1])
    x1_2d = x1.reshape(-1, orig_shape[-1])
    x2_2d = x2.reshape(-1, orig_shape[-1])
    y_2d = torch.empty(x1_2d.shape, dtype=torch.float16, device=x1.device)
    where_kernel(cond_2d, x1_2d, x2_2d, y_2d)
    return y_2d.reshape(orig_shape)
