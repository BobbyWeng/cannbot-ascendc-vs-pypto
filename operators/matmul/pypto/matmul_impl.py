#!/usr/bin/env python3
"""PyPTO MatMul implementation using pypto.frontend.jit + pypto.matmul."""
import os, sys
import torch
import torch_npu
from pypto.frontend import jit
import pypto


@jit
def matmul_kernel(A, B_mat):
    return pypto.matmul(A, B_mat)


def matmul_wrapper(A: torch.Tensor, B_mat: torch.Tensor) -> torch.Tensor:
    """PyPTO JIT-compiled MatMul: Y = A @ B.

    A: [B, 12, 256, 256] FP16
    B: [B, 12, 256, 32] FP16
    Returns: [B, 12, 256, 32] FP16
    """
    return matmul_kernel(A, B_mat)
