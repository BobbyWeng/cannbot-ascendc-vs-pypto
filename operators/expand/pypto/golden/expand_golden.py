import torch

def expand_golden(x: torch.Tensor) -> torch.Tensor:
    return x.expand(x.shape[0], x.shape[1], 384)
