import os, torch, pypto, pypto.op as op

@pypto.frontend.jit
def softmax_kernel_2d(x: pypto.Tensor([], pypto.DT_FP16), y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(1024, 2048)
    max_val = op.amax(x, 1, keepdim=True)
    centered = x - max_val
    exp_vals = op.exp(centered)
    sum_exp = op.sum(exp_vals, 1, keepdim=True)
    y.move(exp_vals / sum_exp)

_cache = {}
def softmax_wrapper(x):
    os = x.shape; k = os
    if k not in _cache:
        x2 = x.reshape(-1, os[-1]); y2 = torch.empty(x2.shape, dtype=torch.float16, device=x.device)
        _cache[k] = (x2, y2)
    else:
        xv, y2 = _cache[k]; x2 = x.reshape(-1, os[-1])
    softmax_kernel_2d(x2, y2)
    return y2.reshape(os)
