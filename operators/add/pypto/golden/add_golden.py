import torch


def add_golden(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference for 4-input Add.
    
    Computation order (fixed left-associative):
        t1 = X1 + X2
        t2 = t1 + X3
        Y  = t2 + X4
    """
    t1 = torch.add(x1, x2)
    t2 = torch.add(t1, x3)
    return torch.add(t2, x4)
