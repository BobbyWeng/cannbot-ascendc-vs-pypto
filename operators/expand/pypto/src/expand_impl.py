import torch
import pypto
import pypto.op


@pypto.frontend.jit
def expand_row(x: pypto.Tensor([1], pypto.DT_FP16),
               y: pypto.Tensor([384], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(128, 1024)
    y.move(pypto.op.expand_clone(x, [384]))


_expand_cache = {}

def expand_wrapper(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    target_last = 384
    x_2d = x.reshape(-1, orig_shape[-1])
    key = orig_shape
    if key not in _expand_cache:
        y_2d = torch.empty(x_2d.shape[0], target_last, dtype=torch.float16, device=x.device)
        _expand_cache[key] = y_2d
    else:
        y_2d = _expand_cache[key]
        if y_2d.shape[0] != x_2d.shape[0]:
            y_2d = torch.empty(x_2d.shape[0], target_last, dtype=torch.float16, device=x.device)
            _expand_cache[key] = y_2d
    x_1d = x_2d.squeeze(-1)
    for i in range(x_2d.shape[0]):
        expand_row(x_1d[i:i+1], y_2d[i])
    return y_2d.reshape(*orig_shape[:-1], target_last)
