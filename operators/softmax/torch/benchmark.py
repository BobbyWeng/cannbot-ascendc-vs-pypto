#!/usr/bin/env python3
import os,sys,json,argparse,numpy as np,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),"common"))
import torch,torch_npu; from benchmark import compute_statistics,compute_effective_bandwidth
SHAPE_TAIL=[256,32]; DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
def run(B,w,l,r,d):
    torch.npu.set_device(d)
    x=torch.from_numpy(np.fromfile(os.path.join(DATA_DIR,f"input_b{B}_fp16.bin"),dtype=np.float16).reshape([B]+SHAPE_TAIL)).npu(d)
    for _ in range(w): torch.nn.functional.softmax(x,dim=-1)
    torch.npu.synchronize(d)
    lats=[]
    for _ in range(r):
        se=torch.npu.Event(enable_timing=True); ee=torch.npu.Event(enable_timing=True)
        se.record()
        for _ in range(l): torch.nn.functional.softmax(x,dim=-1)
        ee.record(); ee.synchronize(); lats.append(se.elapsed_time(ee)*1000.0/l)
    s=compute_statistics(lats)
    tb=B*256*32*2*2; s["effective_bandwidth_gbps"]=round(compute_effective_bandwidth(tb,tb,s["median_us"]),2)
    return {"operator":"softmax","variant":"torch","batch":B,"shape":[B]+SHAPE_TAIL,"dtype":"float16",
            "config":{"warmup":w,"loops":l,"repeat":r},"latency_us":s}
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--batch",default="1,2,4,8,16,32,64")
    p.add_argument("--warmup",type=int,default=200); p.add_argument("--loops",type=int,default=100)
    p.add_argument("--repeat",type=int,default=10); p.add_argument("--device",type=int,default=0)
    a=p.parse_args(); bs=[int(b.strip())for b in a.batch.split(",")]; rs=[]
    for b in bs:
        r=run(b,a.warmup,a.loops,a.repeat,a.device); rs.append(r)
        print(json.dumps(r,indent=2))
    json.dump({"results":rs,"framework":"torch","operator":"softmax"},open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"benchmark_results.json"),"w"),indent=2)
