#!/usr/bin/env python3
"""PyPTO MatMul implementation with dynamic-shape 2D path + static 4D path.

d1c290f36 constraint: cube TileAnnotation requires divisible K/N/M matching tile dims.
2D path uses dynamic shape (pypto.Tensor([], DT_FP16)) for shape flexibility.
4D path keeps static annotation for compatibility.
"""
import os, sys, torch, torch_npu
from pypto.frontend import jit
from pypto.tensor import TensorAnnotation
import pypto

@jit
def matmul_2d_dynamic(A: pypto.Tensor([], pypto.DT_FP16),
                      B: pypto.Tensor([], pypto.DT_FP16),
                      Y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_cube_tile_shapes([16, 32], [16, 32], [16, 32])
    c = pypto.matmul(A, B, pypto.DT_FP16)
    Y.move(c)

A_4d_def = TensorAnnotation((1, 12, 256, 256), pypto.DT_FP16, "A")
B_4d_def = TensorAnnotation((1, 12, 256, 32), pypto.DT_FP16, "B_mat")
Y_4d_def = TensorAnnotation((1, 12, 256, 32), pypto.DT_FP16, "Y")

@jit
def matmul_4d_kernel(A: A_4d_def, B_mat: B_4d_def, Y: Y_4d_def):
    pypto.set_cube_tile_shapes([16, 32], [16, 32], [16, 32])
    c = pypto.matmul(A, B_mat, pypto.DT_FP16)
    Y.move(c)

def matmul_wrapper(A, B_mat):
    ndim = A.dim()
    if ndim == 2:
        Y = torch.empty(A.shape[0], B_mat.shape[1], dtype=torch.float16, device=A.device)
        matmul_2d_dynamic(A, B_mat, Y)
        return Y
    elif ndim == 3:
        Bc, M, K = A.shape
        N = B_mat.shape[-1]
        Y = torch.empty(Bc, M, N, dtype=torch.float16, device=A.device)
        for b in range(Bc):
            matmul_2d_dynamic(A[b], B_mat[b], Y[b])
        return Y
    elif ndim == 4:
        out_shape = list(A.shape[:-1]) + [B_mat.shape[-1]]
        Y = torch.empty(out_shape, dtype=torch.float16, device=A.device)
        matmul_4d_kernel(A, B_mat, Y)
        return Y
    else:
        raise ValueError(f"Unsupported dims: {ndim}")
