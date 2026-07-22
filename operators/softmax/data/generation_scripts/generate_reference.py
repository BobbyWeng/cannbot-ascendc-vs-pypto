import torch, numpy as np, os
import torch_npu
torch.npu.set_device(0)
SEED=20260715; BATCHES=[1,2,4,8,16,32,64]
SHAPE=(256,32); DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"data")
for B in BATCHES:
    x=torch.from_numpy(np.fromfile(os.path.join(DATA_DIR,f"input_b{B}_fp16.bin"),dtype=np.float16).reshape(B,256,32))
    y=torch.nn.functional.softmax(x.npu(),dim=-1).cpu().numpy().astype(np.float16)
    y.tofile(os.path.join(DATA_DIR,f"reference_b{B}_fp16.bin"))
    print(f"B={B}: ref ok")
print("Done")
