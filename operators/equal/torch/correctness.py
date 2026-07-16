#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
import torch
import torch_npu

SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check_correctness_bool(output, reference, label=""):
    if output.shape != reference.shape:
        raise ValueError(f"Shape mismatch: output {tuple(output.shape)} vs reference {tuple(reference.shape)}")
    bitwise_equal = torch.equal(output, reference)
    mismatch_count = int((output != reference).sum().item())
    passed = bitwise_equal
    return {
        "status": "PASS" if passed else "FAIL",
        "bitwise_equal": bitwise_equal,
        "bitwise_mismatch_count": mismatch_count,
        "true_in_output": int(output.sum().item()),
        "true_in_reference": int(reference.sum().item()),
        "label": label,
    }


def run_correctness(batch, device_id):
    torch.npu.set_device(device_id)
    shape = [batch] + SHAPE_TAIL
    x1_path = os.path.join(DATA_DIR, f"x1_b{batch}_fp16.bin")
    x2_path = os.path.join(DATA_DIR, f"x2_b{batch}_fp16.bin")
    ref_path = os.path.join(DATA_DIR, f"reference_b{batch}_bool.bin")

    missing = []
    for p, name in [(x1_path, "x1"), (x2_path, "x2"), (ref_path, "ref")]:
        if not os.path.exists(p):
            missing.append(name)
    if missing:
        return {"batch": batch, "status": "SKIP", "error": f"Missing files: {missing}"}

    x1 = torch.from_numpy(np.fromfile(x1_path, dtype=np.float16).reshape(shape))
    x2 = torch.from_numpy(np.fromfile(x2_path, dtype=np.float16).reshape(shape))
    ref = torch.from_numpy(np.fromfile(ref_path, dtype=np.uint8).reshape(shape))

    x1_npu = x1.npu(device_id)
    x2_npu = x2.npu(device_id)

    output = torch.eq(x1_npu, x2_npu)
    torch.npu.synchronize(device_id)

    output_cpu = output.cpu().bool()
    ref_cpu = ref.bool()

    result = check_correctness_bool(output_cpu, ref_cpu, label=f"B={batch}")
    result["batch"] = batch
    result["shape"] = [batch] + SHAPE_TAIL
    result["dtype_input"] = "float16"
    result["dtype_output"] = "bool"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch baseline correctness for Equal")
    parser.add_argument("--batch", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    args = parser.parse_args()

    batches = [int(b.strip()) for b in args.batch.split(",")]
    all_pass = True
    results = []
    for b in batches:
        result = run_correctness(b, args.device)
        results.append(result)
        status = result["status"]
        if status == "PASS":
            print(f"  B={b}: PASS")
        elif status == "SKIP":
            print(f"  B={b}: SKIP ({result.get('error', '')})")
        else:
            print(f"  B={b}: FAIL (mismatches={result.get('bitwise_mismatch_count', '?')})")
            all_pass = False

    output = {
        "operator": "equal",
        "variant": "torch",
        "results": results,
        "all_pass": all_pass,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.exit(0 if all_pass else 1)
