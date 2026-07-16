"""Pure PyTorch reference implementation for Expand."""
import torch

def expand_golden(x: torch.Tensor) -> torch.Tensor:
    return x.expand(x.shape[0], 256, 384).contiguous()
