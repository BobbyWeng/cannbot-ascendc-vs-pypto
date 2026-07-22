"""Pure PyTorch reference implementation for LayerNorm."""
import torch

def layernorm_golden(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    normalized_shape = weight.shape
    return torch.nn.functional.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)
