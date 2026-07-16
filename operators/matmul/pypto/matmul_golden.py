#!/usr/bin/env python3
"""Golden reference implementation for MatMul.

Exports matmul_golden() for use as baseline reference.
"""
import torch


def matmul_golden(A: torch.Tensor, B_mat: torch.Tensor) -> torch.Tensor:
    """Compute Y = A @ B as a batched matrix multiplication.

    A: [B, H, M, K] FP16
    B: [B, H, K, N] FP16
    Returns: [B, H, M, N] FP16
    """
    return torch.matmul(A.half(), B_mat.half()).half()
