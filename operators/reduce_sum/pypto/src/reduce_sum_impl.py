import torch
import pypto
import pypto.op


@pypto.frontend.jit
def reduce_sum_kernel(x: pypto.Tensor([], pypto.DT_FP16),
                      y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.op.sum(x, dim=-1))


def reduce_sum_wrapper(x: torch.Tensor) -> torch.Tensor:
    """Host wrapper: returns sum(x, dim=-1) using 2D reshape."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    y_2d = torch.empty(x_2d.shape[:-1], dtype=torch.float16, device=x.device)
    reduce_sum_kernel(x_2d, y_2d)
    return y_2d.reshape(orig_shape[:-1])
