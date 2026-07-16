import torch


def not_golden(x: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference for LogicalNot on BOOL (uint8) tensor.
    
    Logical NOT: 0 -> True(1), nonzero -> False(0)
    """
    return torch.logical_not(x.bool()).byte()
