import torch
import pypto.frontend.jit as jit

@jit
def transpose_kernel(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()

def transpose_wrapper(x: torch.Tensor) -> torch.Tensor:
    return transpose_kernel(x)
