import torch

def transpose_golden(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(2, 1)  # [B, H, W] -> [B, W, H]
