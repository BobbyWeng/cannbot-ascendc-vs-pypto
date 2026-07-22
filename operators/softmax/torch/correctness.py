#!/usr/bin/env python3
import os,sys,json,argparse,numpy as np,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch,torch_npu; from common.correctness import check_correctness
SHAPE_TAIL=[256,32]; DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
def run(B,d):
    torch.npu.set_device(d)
    x=torch.from_numpy(np.fromfile(os.path.join(DATA_DIR,f"input_b{B}_fp16.bin"),dtype=np.float16).reshape([B]+SHAPE_TAIL)).npu(d)
    ref=torch.from_numpy(np.fromfile(os.path.join(DATA_DIR,f"reference_b{B}_fp16.bin"),dtype=np.float16).reshape([B]+SHAPE_TAIL)).npu(d)
    o=torch.nn.functional.softmax(x,dim=-1); torch.npu.synchronize(d)
    r=check_correctness(o,ref,rtol=0.001,atol=0.01,require_bitwise=False,label=f"B={B}")
    r["batch"]=B; r["shape"]=[B]+SHAPE_TAIL; r["dtype"]="float16"; return r
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--batch",default="1,2,4,8,16,32,64"); p.add_argument("--device",type=int,default=0)
    a=p.parse_args(); bs=[int(b.strip())for b in a.batch.split(",")]; ap=True; rs=[]
    for b in bs:
        r=run(b,a.device); rs.append(r); s=r["status"]
        if s=="PASS": print(f"  B={b}: PASS")
        else: print(f"  B={b}: FAIL"); ap=False
    o={"operator":"softmax","variant":"torch","results":rs,"all_pass":ap}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"correctness_results.json"),"w") as f: json.dump(o,f,indent=2)
    sys.exit(0 if ap else 1)
