import torch


def equal_golden(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference for element-wise Equal.

    Returns BOOL tensor: True where x1 == x2, False elsewhere.
    NaN semantics: NaN != NaN (consistent with torch.eq).
    """
    return torch.eq(x1, x2)
