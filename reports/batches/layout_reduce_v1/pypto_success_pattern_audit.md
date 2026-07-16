# PyPTO Success Pattern Audit

## Common JIT Pattern (ReLU, Mul, Add, Not, Or)

### 1. File Organization
```
operators/{op}/pypto/
├── src/
│   ├── {op}_impl.py          # JIT function + wrapper (top-level, non-nested)
│   └── __init__.py (optional — 3/5 have it, 2/5 don't; both work)
├── tests/
│   └── test_{op}.py          # Standard import + test
├── golden/
│   └── {op}_golden.py        # Pure PyTorch reference
└── ... spec/design/api reports
```

### 2. JIT Function Rules (verified working)

| Rule | Value |
|------|-------|
| JIT decorator | `@pypto.frontend.jit` (no parenthesized call, no options) |
| Location | Top-level in `{op}_impl.py` only |
| Nested functions | NEVER |
| Lambda | NEVER |
| Heredoc / __main__ / python -c | NEVER |
| Wrapper function | Always a separate pure-Python function |
| Re-exports in __init__.py | Optional, wrapper imported via `from {op}_impl import {wrapper}` |
| sys.path.insert | 2 inserts: `'..', 'golden'` and `'..', 'src'` (relative to test file) |

### 3. Function Signature Pattern (all successful)

```python
@pypto.frontend.jit
def {op}_kernel_2d(x: pypto.Tensor([], pypto.DT_FP16),   # or DT_BOOL/DT_UINT8
                   y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(..., ...)
    y.move(pypto.op.{op}(x))          # for unary
    # or: y.move(pypto.op.{op}(x1, x2))  for binary
```

### 4. Wrapper Pattern (all successful)

```python
def {op}_wrapper(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    y_2d = torch.empty(x_2d.shape, dtype=torch.float16, device=x.device)
    {op}_kernel_2d(x_2d, y_2d)       # or (x1_2d, x2_2d, y_2d) for binary
    return y_2d.reshape(orig_shape)
```

### 5. Test Pattern (all successful)

```python
# Setup:
sys.path.insert(0, os.path.join(test_dir, '..', 'golden'))
sys.path.insert(0, os.path.join(test_dir, '..', 'src'))
from {op}_golden import {op}_golden
from {op}_impl import {op}_wrapper

# Device:
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
import torch_npu
torch.npu.set_device(device_id)

# Call:
y = {op}_wrapper(x_npu)
actual = y.cpu().to(torch.float16)
```

### 6. pypto.frontend.jit Source Retrieval Mechanism

The Source class in `pypto/frontend/parser/diagnostics.py` calls:
```python
source_lines, start_line = inspect.getsourcelines(program)
```

This requires:
- `program.__code__` exists (function is a real Python function, not lambda/nested/live)
- The source file is findable via `inspect.getfile()` or `__code__.co_filename`
- The source file is still present on disk

### 7. What `inspect.getsourcelines` needs

- A real `function` object (not a callable class, not lambda, not closure)
- The original `.py` file must exist and be readable
- The function must be defined at module level (top-level def)
- No `importlib` reload tricks after JIT wrapping

### 8. Key Finding

**`__init__.py` in src/ is NOT required** — 2/5 successful operators (relu, mul) have NO `__init__.py`, yet they work. The import goes through `sys.path.insert` + direct module name, not package-qualified path.

The JIT source retrieval failure ("JIT cannot get source code") is NOT caused by missing `__init__.py`. It must be caused by:
1. Empty src/ directories (as in Expand, Transpose currently)
2. Or function defined in wrong scope (closure, lambda, etc.)
3. Or import mechanism that hides the true source file
