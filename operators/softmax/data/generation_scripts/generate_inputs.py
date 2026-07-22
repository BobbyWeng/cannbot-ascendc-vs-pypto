import numpy as np, os

SEED=20260715; rng=np.random.default_rng(SEED)
BATCHES=[1,2,4,8,16,32,64]; SHAPE=[256,32]
DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"data")
os.makedirs(DATA_DIR,exist_ok=True)

def gen(batch,path,rs=None):
    if rs is not None: rng=np.random.default_rng(rs)
    s=[batch]+SHAPE; n=int(np.prod(s))
    vals=rng.uniform(-10,10,n).astype(np.float16).reshape(s)
    vals.tofile(path)
    return vals

if __name__=="__main__":
    for b in BATCHES:
        p=os.path.join(DATA_DIR,f"input_b{b}_fp16.bin")
        gen(b,p,SEED+b)
        print(f"  b={b}: ok")
    print("All inputs generated.")
