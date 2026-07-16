"""Scan different tile shapes for (256, 384) binary add"""
import subprocess
import sys
import os
import tempfile

src_dir = '/mnt/workspace/cannbot_ascendc_vs_pypto/operators/add/pypto/src'
impl_path = os.path.join(src_dir, 'add_impl.py')
rows, cols = 256, 384

configs = [
    (384, 256),
    (384, 1024),
    (96, 1024),
    (128, 1024),
    (256, 1024),
    (768, 1024),
    (512, 512),
    (1024, 2048),
    (2048, 1024),
]

for tile_a, tile_b in configs:
    # Write the kernel module source
    with open(impl_path, 'w') as f:
        f.write(f"""import torch
import pypto
import pypto.op

@pypto.frontend.jit
def add_binary_kernel(x1: pypto.Tensor([], pypto.DT_FP16),
                      x2: pypto.Tensor([], pypto.DT_FP16),
                      y: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes({tile_a}, {tile_b})
    y.move(pypto.op.add(x1, x2))

def add_binary(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x1)
    add_binary_kernel(x1, x2, y)
    return y
""")
    
    # Write test script
    test_code = f"""import sys
sys.path.insert(0, '{src_dir}')
import torch, torch_npu, warnings
warnings.filterwarnings("ignore")
torch.npu.set_device(0)

# Clean cache
import glob, os
for f in glob.glob('/tmp/.99*'):
    try:
        os.unlink(f)
    except:
        pass

import importlib
import add_impl
importlib.reload(add_impl)
from add_impl import add_binary_kernel

x1 = torch.randn({rows}, {cols}, dtype=torch.float16).npu(0)
x2 = torch.randn({rows}, {cols}, dtype=torch.float16).npu(0)
y = torch.empty({rows}, {cols}, dtype=torch.float16).npu(0)
try:
    add_binary_kernel(x1, x2, y)
    torch.npu.synchronize(0)
    match = torch.equal(y.cpu(), x1.cpu()+x2.cpu())
    print(f"RESULT: PASS match={{match}}")
except Exception as e:
    print(f"RESULT: FAIL {{type(e).__name__}}: {{str(e)[:150]}}")
"""
    
    with tempfile.NamedTemporaryFile(suffix='.py', prefix='tile_test_', dir='/tmp', delete=False, mode='w') as tf:
        tf.write(test_code)
        tf_path = tf.name
    
    result = subprocess.run(
        ['python3', tf_path],
        capture_output=True, text=True,
        env={**os.environ, 'TILE_FWK_DEVICE_ID': '0'},
        timeout=120
    )
    
    os.unlink(tf_path)
    
    for line in result.stdout.split('\n'):
        if line.startswith('RESULT:'):
            print(f'tile=({tile_a:5d},{tile_b:5d}) -> {line.split("RESULT:")[1].strip()}')
            break
    else:
        err = result.stderr.strip().split('\n')[-1][:100] if result.stderr.strip() else 'no output'
        print(f'tile=({tile_a:5d},{tile_b:5d}) -> ERROR: {err}')
