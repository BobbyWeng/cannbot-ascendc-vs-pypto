import numpy as np
import os
import hashlib
import json

SEED = 20260715
BATCHES = [1, 2, 4, 8, 16, 32, 64]
SHAPE_TAIL = [256, 384]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
os.makedirs(DATA_DIR, exist_ok=True)


def _fp16_bits(x):
    return np.float16(x).view(np.uint16)


FP16_NEG_ZERO = _fp16_bits(-0.0)
FP16_POS_ZERO = _fp16_bits(0.0)
FP16_NAN_1 = _fp16_bits(np.nan)
FP16_NAN_2 = np.uint16(0x7E00)
FP16_POS_INF = _fp16_bits(np.inf)
FP16_NEG_INF = _fp16_bits(-np.inf)
FP16_MAX = _fp16_bits(65504.0)
FP16_MIN_POS = _fp16_bits(6e-8)
FP16_MIN_NEG = _fp16_bits(-6e-8)
FP16_MAX_NEG = _fp16_bits(-65504.0)
FP16_ADJ_1 = _fp16_bits(1.0)
FP16_ADJ_2 = _fp16_bits(1.0009765625)


def build_special_positions(total_elems):
    positions = []
    special_values = []

    identical_count = min(64, total_elems // 16)
    for i in range(identical_count):
        positions.append(i)
        v = np.float16(np.random.uniform(-10, 10))
        special_values.append(("identical", v, v))

    offset_identical = identical_count
    diff_count = min(64, total_elems // 16)
    for i in range(diff_count):
        positions.append(offset_identical + i)
        special_values.append(("different", np.float16(np.random.uniform(-10, 10)), np.float16(np.random.uniform(-10, 10))))

    offset_diff = offset_identical + diff_count
    partial_count = min(64, total_elems // 16)
    for i in range(partial_count):
        positions.append(offset_diff + i)
        if i % 2 == 0:
            special_values.append(("partial_equal", np.float16(3.0), np.float16(3.0)))
        else:
            special_values.append(("partial_unequal", np.float16(5.0), np.float16(-5.0)))

    offset_partial = offset_diff + partial_count

    pos0 = offset_partial
    positions.append(pos0)
    special_values.append(("pos0_neg0", np.float16(0.0), np.float16(-0.0)))

    pos_nan_nan = pos0 + 1
    positions.append(pos_nan_nan)
    special_values.append(("nan_nan", np.float16(np.nan), np.float16(np.nan)))

    pos_nan_normal = pos_nan_nan + 1
    positions.append(pos_nan_normal)
    special_values.append(("nan_normal", np.float16(np.nan), np.float16(1.0)))

    pos_inf_inf = pos_nan_normal + 1
    positions.append(pos_inf_inf)
    special_values.append(("inf_inf", np.float16(np.inf), np.float16(np.inf)))

    pos_inf_neginf = pos_inf_inf + 1
    positions.append(pos_inf_neginf)
    special_values.append(("inf_neginf", np.float16(np.inf), np.float16(-np.inf)))

    pos_adj_1 = pos_inf_neginf + 1
    positions.append(pos_adj_1)
    special_values.append(("adjacent_1", np.float16(1.0), np.float16(1.0009765625)))

    pos_adj_2 = pos_adj_1 + 1
    positions.append(pos_adj_2)
    special_values.append(("adjacent_2", np.float16(1.0009765625), np.float16(1.0)))

    pos_min_max = pos_adj_2 + 1
    positions.append(pos_min_max)
    special_values.append(("min_max", np.float16(6e-8), np.float16(-65504.0)))

    curr_pos = pos_min_max + 1
    pos_zero_nan = curr_pos
    positions.append(pos_zero_nan)
    special_values.append(("zero_nan", np.float16(0.0), np.float16(np.nan)))

    remainder = total_elems - len(positions)
    if remainder > 0:
        rand_x1 = np.random.uniform(-100, 100, remainder).astype(np.float16)
        rand_x2 = np.random.uniform(-100, 100, remainder).astype(np.float16)
        for i in range(remainder):
            positions.append(curr_pos + 1 + i)
            special_values.append(("random", rand_x1[i], rand_x2[i]))

    return positions, special_values


def generate_inputs(batch):
    shape = [batch] + SHAPE_TAIL
    total_elems = int(np.prod(shape))

    x1 = np.zeros(total_elems, dtype=np.float16)
    x2 = np.zeros(total_elems, dtype=np.float16)

    positions, special_values = build_special_positions(total_elems)

    for pos, sv in zip(positions, special_values):
        if pos >= total_elems:
            break
        _, v1, v2 = sv
        x1[pos] = v1
        x2[pos] = v2

    return x1.reshape(shape), x2.reshape(shape)


def compute_reference(x1, x2):
    return (x1 == x2)


def generate_boundary_cases(batch):
    shape = [batch] + SHAPE_TAIL
    total_elems = int(np.prod(shape))
    cases = {}

    all_ones = np.ones(total_elems, dtype=np.float16).reshape(shape)
    all_neg = np.ones(total_elems, dtype=np.float16).reshape(shape) * -1.5
    all_zeros = np.zeros(total_elems, dtype=np.float16).reshape(shape)

    cases["identical_all_ones"] = (all_ones.copy(), all_ones.copy())
    cases["identical_all_neg"] = (all_neg.copy(), all_neg.copy())
    cases["identical_all_zero"] = (all_zeros.copy(), all_zeros.copy())

    more_neg = np.ones(total_elems, dtype=np.float16).reshape(shape) * -2.0
    cases["all_diff"] = (all_ones.copy(), more_neg.copy())

    x1_pos = all_ones.copy()
    x2_neg = np.ones(total_elems, dtype=np.float16).reshape(shape) * -1.0
    cases["pos_vs_neg"] = (x1_pos, x2_neg)

    x1_nan = np.full(total_elems, np.nan, dtype=np.float16).reshape(shape)
    cases["all_nan_vs_nan"] = (x1_nan.copy(), x1_nan.copy())

    x2_normal = np.ones(total_elems, dtype=np.float16).reshape(shape)
    cases["all_nan_vs_normal"] = (x1_nan.copy(), x2_normal.copy())

    x1_inf = np.full(total_elems, np.inf, dtype=np.float16).reshape(shape)
    cases["all_inf_vs_inf"] = (x1_inf.copy(), x1_inf.copy())

    x2_neginf = np.full(total_elems, -np.inf, dtype=np.float16).reshape(shape)
    cases["all_inf_vs_neginf"] = (x1_inf.copy(), x2_neginf.copy())

    return cases


def main():
    print(f"Generating EQUAL test data with seed={SEED}")
    manifest = {
        "operator": "equal",
        "seed": SEED,
        "dtype_input": "float16",
        "dtype_output": "bool",
        "shape": "[B, 256, 384]",
        "element_count_per_batch": 256 * 384,
        "byte_size_per_input_per_batch": 256 * 384 * 2,
        "byte_size_per_output_per_batch": 256 * 384 * 1,
        "batches": BATCHES,
        "boundary_cases": [
            "identical_all_ones", "identical_all_neg", "identical_all_zero",
            "all_diff", "pos_vs_neg",
            "all_nan_vs_nan", "all_nan_vs_normal",
            "all_inf_vs_inf", "all_inf_vs_neginf"
        ],
        "input_files": [],
        "reference_files": []
    }

    for b in BATCHES:
        x1, x2 = generate_inputs(b)
        ref = compute_reference(x1, x2)

        x1_path = os.path.join(DATA_DIR, f"x1_b{b}_fp16.bin")
        x2_path = os.path.join(DATA_DIR, f"x2_b{b}_fp16.bin")
        ref_path = os.path.join(DATA_DIR, f"reference_b{b}_bool.bin")

        x1.tofile(x1_path)
        x2.tofile(x2_path)
        ref.tofile(ref_path)

        sha1 = hashlib.sha256()
        with open(x1_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha1.update(chunk)
        sha2 = hashlib.sha256()
        with open(x2_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha2.update(chunk)
        shar = hashlib.sha256()
        with open(ref_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                shar.update(chunk)

        manifest["input_files"].append({
            "name": f"x1_b{b}_fp16.bin",
            "batch": b,
            "input_index": 1,
            "shape": f"[{b}, 256, 384]",
            "dtype": "float16",
            "element_count": b * 256 * 384,
            "byte_size": b * 256 * 384 * 2,
            "seed": SEED,
            "sha256": sha1.hexdigest()
        })
        manifest["input_files"].append({
            "name": f"x2_b{b}_fp16.bin",
            "batch": b,
            "input_index": 2,
            "shape": f"[{b}, 256, 384]",
            "dtype": "float16",
            "element_count": b * 256 * 384,
            "byte_size": b * 256 * 384 * 2,
            "seed": SEED,
            "sha256": sha2.hexdigest()
        })
        manifest["reference_files"].append({
            "name": f"reference_b{b}_bool.bin",
            "batch": b,
            "shape": f"[{b}, 256, 384]",
            "dtype": "bool",
            "element_count": b * 256 * 384,
            "byte_size": b * 256 * 384 * 1,
            "seed": SEED,
            "computation_order": "torch.eq(X1, X2) in FP16 -> BOOL",
            "sha256": shar.hexdigest()
        })

        boundary_cases = generate_boundary_cases(b)
        for case_name, (bx1, bx2) in boundary_cases.items():
            bref = compute_reference(bx1, bx2)
            bx1_path = os.path.join(DATA_DIR, f"x1_b{b}_{case_name}.bin")
            bx2_path = os.path.join(DATA_DIR, f"x2_b{b}_{case_name}.bin")
            bref_path = os.path.join(DATA_DIR, f"reference_b{b}_{case_name}.bin")
            bx1.tofile(bx1_path)
            bx2.tofile(bx2_path)
            bref.tofile(bref_path)

        print(f"  B={b}: {x1.shape}, special positions used")

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")
    print("All EQUAL inputs generated successfully.")


if __name__ == "__main__":
    main()
