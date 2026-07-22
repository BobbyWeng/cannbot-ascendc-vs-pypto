import torch
def softmax_golden(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)
