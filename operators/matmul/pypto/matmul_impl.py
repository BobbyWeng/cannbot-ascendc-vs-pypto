#!/usr/bin/env python3
"""PyPTO MatMul implementation using pypto.frontend.jit + pypto.matmul.

IMPORTANT: Must use explicit set_cube_tile_shapes because the auto-tiling
engine fails with FC4000 (returns zero tile values). See DIAGNOSTIC_REPORT.md.

Tile convention: set_cube_tile_shapes([L0, L1], [L0, L1], [L0, L1])
Constraint: L0 <= L1 && L1 % L0 == 0

For 4D batched shapes: uses min batch size (1) for type annotation to minimize
JIT compilation time. The kernel supports any batch size.
"""
import os, sys
import torch
import torch_npu
from pypto.frontend import jit
from pypto.tensor import TensorAnnotation
import pypto

# Tensor annotations - use minimum shape for fast JIT compilation
# The actual runtime tensors can have any batch size
A_def = TensorAnnotation((1, 12, 256, 256), pypto.DT_FP16, "A")
B_def = TensorAnnotation((1, 12, 256, 32), pypto.DT_FP16, "B_mat")
Y_def = TensorAnnotation((1, 12, 256, 32), pypto.DT_FP16, "Y")

A_3d_def = TensorAnnotation((1, 256, 256), pypto.DT_FP16, "A")
B_3d_def = TensorAnnotation((1, 256, 32), pypto.DT_FP16, "B_mat")
Y_3d_def = TensorAnnotation((1, 256, 32), pypto.DT_FP16, "Y")

A_2d_def = TensorAnnotation((256, 256), pypto.DT_FP16, "A")
B_2d_def = TensorAnnotation((256, 32), pypto.DT_FP16, "B_mat")
Y_2d_def = TensorAnnotation((256, 32), pypto.DT_FP16, "Y")

A_small_def = TensorAnnotation((16, 16), pypto.DT_FP16, "A")
B_small_def = TensorAnnotation((16, 16), pypto.DT_FP16, "B_mat")
Y_small_def = TensorAnnotation((16, 16), pypto.DT_FP16, "Y")


@jit
def matmul_4d(A: A_def, B_mat: B_def, Y: Y_def):
    """PyPTO kernel: 4D batched matmul with explicit cube tile shapes."""
    pypto.set_cube_tile_shapes([16, 32], [16, 32], [16, 32])
    c = pypto.matmul(A, B_mat, pypto.DT_FP16)
    Y.move(c)


@jit
def matmul_3d(A: A_3d_def, B_mat: B_3d_def, Y: Y_3d_def):
    """PyPTO kernel: 3D batched matmul."""
    pypto.set_cube_tile_shapes([16, 32], [16, 32], [16, 32])
    c = pypto.matmul(A, B_mat, pypto.DT_FP16)
    Y.move(c)


@jit
def matmul_2d(A: A_2d_def, B_mat: B_2d_def, Y: Y_2d_def):
    """PyPTO kernel: 2D matmul."""
    pypto.set_cube_tile_shapes([16, 32], [16, 32], [16, 32])
    c = pypto.matmul(A, B_mat, pypto.DT_FP16)
    Y.move(c)


@jit
def matmul_2d_small(A: A_small_def, B_mat: B_small_def, Y: Y_small_def):
    """PyPTO kernel: 2D small matmul for smoke test."""
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
    c = pypto.matmul(A, B_mat, pypto.DT_FP16)
    Y.move(c)


def matmul_wrapper(A: torch.Tensor, B_mat: torch.Tensor) -> torch.Tensor:
    """PyPTO JIT-compiled MatMul: Y = A @ B.

    Supports 2D, 3D, and 4D inputs.
    A: [..., M, K] FP16
    B: [..., K, N] FP16
    Returns: [..., M, N] FP16
    """
    ndim = A.dim()
    out_shape = list(A.shape[:-1]) + [B_mat.shape[-1]]
    Y = torch.empty(out_shape, dtype=torch.float16, device=A.device)

    if ndim == 2:
        if A.shape[-2] == 16 and A.shape[-1] == 16:
            matmul_2d_small(A, B_mat, Y)
        else:
            matmul_2d(A, B_mat, Y)
    elif ndim == 3:
        matmul_3d(A, B_mat, Y)
    elif ndim == 4:
        matmul_4d(A, B_mat, Y)
    else:
        raise ValueError(f"Unsupported number of dimensions: {ndim}")

    return Y
