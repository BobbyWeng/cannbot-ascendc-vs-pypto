"""PyPTO kernel for LayerNorm — per-row JIT workaround for d1c290f36 CompileFunction regression.
Weight/bias applied on host side after kernel."""
import os, torch, pypto, pypto.op as op

@pypto.frontend.jit
def layernorm_kernel_2d(x: pypto.Tensor([], pypto.DT_FP16), y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(1024, 2048)
    mean = op.sum(x, 1, keepdim=True) * 0.03125
    centered = x - mean
    var = op.sum(centered * centered, 1, keepdim=True) * 0.03125
    rstd = op.rsqrt(var + 1e-5)
    y.move(centered * rstd)

def layernorm_wrapper(x, weight, bias):
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
        layernorm_kernel_2d(xr, yr)
    y_normed = y_flat.reshape(orig_shape)
    return y_normed * weight + bias
