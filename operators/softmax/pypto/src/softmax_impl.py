"""PyPTO kernel for Softmax — per-row JIT workaround for d1c290f36 CompileFunction regression."""
import os, torch, pypto, pypto.op as op

@pypto.frontend.jit
def softmax_kernel_2d(x: pypto.Tensor([], pypto.DT_FP16), y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(1024, 2048)
    max_val = op.amax(x, 1, keepdim=True)
    centered = x - max_val
    exp_vals = op.exp(centered)
    sum_exp = op.sum(exp_vals, 1, keepdim=True)
    y.move(exp_vals / sum_exp)

def softmax_wrapper(x):
    orig_shape = x.shape
    batch_size = orig_shape[0]
    row_count = orig_shape[1:].numel() // orig_shape[-1] if len(orig_shape) >= 3 else 1
    last_dim = orig_shape[-1]
    total_rows = batch_size * row_count
    x_flat = x.reshape(total_rows, last_dim)
    y_flat = torch.empty(total_rows, last_dim, dtype=torch.float16, device=x.device)
    for r in range(total_rows):
        xr = x_flat[r].reshape(1, last_dim)
        yr = y_flat[r].reshape(1, last_dim)
        softmax_kernel_2d(xr, yr)
    return y_flat.reshape(orig_shape)
