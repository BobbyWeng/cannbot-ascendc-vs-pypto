import torch


def or_golden(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference for Logical OR on BOOL (uint8) tensors."""
    return torch.logical_or(x1, x2).to(torch.uint8)
