"""Pure PyTorch reference implementation for ReLU."""
import torch

def relu_golden(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)
