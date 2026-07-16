import torch


def where_golden(condition: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference for Where.

    Y[i] = Condition[i] ? X1[i] : X2[i]
    NaN/Inf in non-selected branch must NOT propagate.
    """
    return torch.where(condition.bool(), x1, x2)
