import torch


def reduce_sum_golden(x: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference for ReduceSum.
    Computes sum over last dimension.
    """
    return torch.sum(x, dim=-1)
