import torch
import pypto


@pypto.frontend.jit
def equal_kernel(x1: pypto.Tensor([], pypto.DT_FP16),
                 x2: pypto.Tensor([], pypto.DT_FP16),
                 y: pypto.Tensor([], pypto.DT_BOOL)):
    pypto.set_vec_tile_shapes(64, 1024)
    y.move(pypto.eq(x1, x2))


def equal_wrapper(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    orig_shape = x1.shape
    x1_2d = x1.reshape(-1, orig_shape[-1])
    x2_2d = x2.reshape(-1, orig_shape[-1])
    y_2d = torch.empty(x1_2d.shape, dtype=torch.bool, device=x1.device)
    equal_kernel(x1_2d, x2_2d, y_2d)
    return y_2d.reshape(orig_shape)


equal_wrapper = equal_wrapper
