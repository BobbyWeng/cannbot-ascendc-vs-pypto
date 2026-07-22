import os, sys, torch, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','golden'))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','src'))
from softmax_golden import softmax_golden; from softmax_impl import softmax_wrapper
BS=[1,2,4,8,16,32,64]; SH=(256,32); DT=torch.float16; SEED=20260715
def test():
    torch.manual_seed(SEED); did=int(os.environ.get("TILE_FWK_DEVICE_ID","0"))
    import torch_npu; torch.npu.set_device(did); ap=True
    for B in BS:
        x=torch.randn((B,)+SH,dtype=DT); xn=x.npu(); exp=softmax_golden(x)
        y=softmax_wrapper(xn); act=y.cpu().to(DT)
        d=(act.float()-exp.float()).abs(); rd=d/(exp.float().abs()+1e-12)
        mx=d.max().item(); mr=rd.max().item(); mm=int((d>0.01).sum().item()) if mx>0.01 else 0
        pas=mx<=0.01 or mr<=0.001
        print(f"[{'PRECISION_PASS' if pas else 'PRECISION_FAIL'}] B={B}: max_abs={mx:.6f} max_rel={mr:.6f}")
        if not pas: ap=False
    if ap: print("[PRECISION_PASS] All done")
    else: print("[PRECISION_FAIL] Some failed"); sys.exit(1)
if __name__=="__main__": test()
