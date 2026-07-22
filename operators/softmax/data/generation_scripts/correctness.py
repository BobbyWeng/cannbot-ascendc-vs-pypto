import numpy as np, hashlib, os
SHAPE=[256,32]
def check(output_path,ref_path,batch,rtol=0.001,atol=0.01,verbose=True):
    o=np.fromfile(output_path,dtype=np.float16); r=np.fromfile(ref_path,dtype=np.float16)
    n=int(np.prod([batch]+SHAPE))
    if o.size!=r.size or o.size!=n: return {"status":"FAIL","error":"size"}
    d=np.abs(o.astype(np.float32)-r.astype(np.float32)); rd=d/(np.abs(r.astype(np.float32))+1e-12)
    md=float(d.max()); mr=float(rd.max())
    passed=(md<=atol or mr<=rtol) and np.isnan(o).sum()==0 and np.isinf(o).sum()==0
    res={"status":"PASS" if passed else "FAIL","total":n,"max_abs":md,"max_rel":mr,
         "nan":int(np.isnan(o).sum()),"inf":int(np.isinf(o).sum())}
    if verbose: print(f"  {res['status']}: max_abs={md:.6e}")
    return res
if __name__=="__main__":
    import sys
    r=check(sys.argv[1],sys.argv[2],int(sys.argv[3]))
    sys.exit(0 if r["status"]=="PASS" else 1)
