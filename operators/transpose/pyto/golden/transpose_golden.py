import torch

def transpose_golden(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()
