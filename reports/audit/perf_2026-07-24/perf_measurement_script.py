import subprocess, os, json, time

BASE = '/mnt/workspace/gitCode/cann/pto-isa/cannbot-ascendc-vs-pypto'
PYPTO = '/home/developer/.cannbot/repo/plugins-official/pypto-op-orchestrator/pypto/python'
BATCHES = [1, 8, 64]
WARMUP = 100
ITERS = 50

def make_script(op, B):
    imp = "sys.path[:0]=['"+PYPTO+"']; import torch, torch_npu; torch.npu.set_device(0); import importlib.util, time"
    
    configs = {
        'add': (f"{BASE}/operators/add/pypto/src/add_impl.py", "add_binary",
                f"x=torch.randn({B},12,256,256,dtype=torch.float16).npu(); y=torch.randn({B},12,256,256,dtype=torch.float16).npu()",
                "w(x,y)"),
        'where': (f"{BASE}/operators/where/pypto/src/where_impl.py", "where_wrapper",
                f"c=(torch.rand({B},65536)>0.5).npu(); xx=torch.randn({B},65536,dtype=torch.float16).npu(); yy=torch.randn({B},65536,dtype=torch.float16).npu()",
                "w(c,xx,yy)"),
        'transpose': (f"{BASE}/operators/transpose/pypto/src/transpose_impl.py", "transpose_wrapper",
                f"x=torch.randn({B},256,384,dtype=torch.float16).npu()",
                "w(x)"),
        'relu': (f"{BASE}/operators/relu/pypto/src/relu_impl.py", "relu_wrapper",
                f"x=torch.randn({B},65536,dtype=torch.float16).npu()",
                "w(x)"),
        'mul': (f"{BASE}/operators/mul/pypto/src/mul_impl.py", "mul_wrapper",
                f"x=torch.randn({B},65536,dtype=torch.float16).npu(); y=torch.randn({B},65536,dtype=torch.float16).npu()",
                "w(x,y)"),
        'softmax': (f"{BASE}/operators/softmax/pypto/src/softmax_impl.py", "softmax_wrapper",
                f"x=torch.randn({B*256},384,dtype=torch.float16).npu()",
                "w(x)"),
        'layernorm': (f"{BASE}/operators/layernorm/pypto/src/layernorm_impl.py", "layernorm_wrapper",
                f"x=torch.randn({B},256,32,dtype=torch.float16).npu(); wb=torch.ones(32,dtype=torch.float16).npu(); bb=torch.zeros(32,dtype=torch.float16).npu()",
                "w(x,wb,bb)"),
        'matmul': (f"{BASE}/operators/matmul/pypto/matmul_impl.py", "matmul_wrapper",
                f"A=torch.randn({B},256,384,dtype=torch.float16).npu(); Bt=torch.randn({B},384,512,dtype=torch.float16).npu()",
                "w(A,Bt)"),
    }
    
    if op not in configs:
        return None
    
    impl_path, wfn, setup, call = configs[op]
    
    script = f"""import sys
{imp}
spec=importlib.util.spec_from_file_location('m',r'{impl_path}')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
w=mod.{wfn}
{setup}
{call}; torch.npu.synchronize()
for _ in range({WARMUP}): {call}; torch.npu.synchronize()
t0=time.time()
for _ in range({ITERS}): {call}; torch.npu.synchronize()
t1=time.time()
print(f"OK|{{((t1-t0)*1e6/{ITERS}):.1f}}")
"""
    return script

results = {}
ops_list = ['add','where','transpose','relu','mul','softmax','layernorm','matmul']

for op in ops_list:
    print(f"\n=== {op} ===", flush=True)
    results[op] = {}
    for B in BATCHES:
        script = make_script(op, B)
        if script is None:
            continue
        try:
            r = subprocess.run(['python3','-c',script], capture_output=True, text=True, timeout=300)
            for line in r.stdout.split('\n'):
                if line.startswith('OK|'):
                    us = float(line.split('|')[1])
                    results[op][B] = {'host_us': us}
                    print(f"  B={B:>3}: {us:>10.1f} us", flush=True)
                    break
            else:
                err = (r.stderr or '')[-150:]
                results[op][B] = {'error': 'no output', 'stderr': err[:120]}
                print(f"  B={B:>3}: FAIL {err[:80]}", flush=True)
        except subprocess.TimeoutExpired:
            results[op][B] = {'error': 'timeout'}
            print(f"  B={B:>3}: TIMEOUT", flush=True)
        except Exception as e:
            results[op][B] = {'error': str(e)[:100]}
            print(f"  B={B:>3}: ERROR", flush=True)

outdir = '/tmp/opencode/host_timing'
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/results3.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Print summary table
print("\n\n=== SUMMARY: host_synchronized_operation_us ===")
header = f"{'Op':<12}" + "".join(f"|{'B='+str(b):>12}" for b in BATCHES)
print(header)
print("-" * len(header))
for op in ops_list:
    row = f"{op:<12}"
    for B in BATCHES:
        v = results.get(op,{}).get(B,{})
        if 'host_us' in v:
            row += f"|{v['host_us']:>12.1f}"
        else:
            row += f"|{'FAIL':>12}"
    print(row)
