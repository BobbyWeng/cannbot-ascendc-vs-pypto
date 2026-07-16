"""Pure PyTorch reference implementation for Mul."""
import torch

def mul_golden(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.mul(x1, x2)
