"""Pure PyTorch reference implementation for broadcast Div."""
import torch


def div_golden(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Broadcast division: Y = X1 / X2 (broadcast along last dim).

    X2 must have shape [B, 12, 256, 1] broadcastable to [B, 12, 256, 256].
    """
    return torch.div(x1, x2)
