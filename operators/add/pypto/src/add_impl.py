import torch
import pypto
import pypto.op


@pypto.frontend.jit
def add_binary_kernel(x1: pypto.Tensor([], pypto.DT_FP16),
                      x2: pypto.Tensor([], pypto.DT_FP16),
                      y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.op.add(x1, x2))


def add_binary(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Host wrapper: returns x1 + x2 using 2D reshape."""
    orig_shape = x1.shape
    x1_2d = x1.reshape(-1, orig_shape[-1])
    x2_2d = x2.reshape(-1, orig_shape[-1])
    y_2d = torch.empty(x1_2d.shape, dtype=torch.float16, device=x1.device)
    add_binary_kernel(x1_2d, x2_2d, y_2d)
    return y_2d.reshape(orig_shape)


def add_4(x1: torch.Tensor, x2: torch.Tensor,
          x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
    """Four-input Add via three chained binary adds: ((x1+x2)+x3)+x4"""
    t = add_binary(x1, x2)
    t = add_binary(t, x3)
    return add_binary(t, x4)


# Unified entry point alias
add_wrapper = add_4
