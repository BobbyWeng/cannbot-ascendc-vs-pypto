#!/usr/bin/env python3
"""Rebuild final comparison CSVs and dashboard from parsed msprof data."""
import json, os, csv, glob
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPERATORS_DIR = os.path.join(PROJECT_ROOT, "operators")

OPERATOR_META = {
    "relu":     {"shape": "[B,12,256,32]",           "dtype": "FP16", "formula": "Y = max(X, 0)", "batches": [1,2,4,8,16,32,64], "precision": "位精确", "status": "COMPLETE"},
    "mul":      {"shape": "[B,3,4,256,32]",          "dtype": "FP16", "formula": "Y = X1 * X2", "batches": [1,2,4,8,16,32,64], "precision": "位精确", "status": "COMPLETE"},
    "add":      {"shape": "[B,256,384]",              "dtype": "FP16", "formula": "Y = (((X1+X2)+X3)+X4)", "batches": [1,2,4,8,16,32,64], "precision": "位精确", "status": "COMPLETE_WITH_LIMITATION"},
    "div":      {"shape": "X1[B,12,256,256], X2[B,12,256,1]", "dtype": "FP16", "formula": "Y = X1 / X2 (广播)", "batches": [1,2,4,8,16,32,64], "precision": "位精确", "status": "COMPLETE_WITH_LIMITATION"},
    "equal":    {"shape": "[B,12,256,256] (bool输出)", "dtype": "FP16→bool", "formula": "Y = (X1 == X2)", "batches": [1,2,4,8,16,32,64], "precision": "位精确(bool)", "status": "COMPLETE_WITH_LIMITATION"},
    "not":      {"shape": "[B,12,256,256] (bool)",    "dtype": "Bool", "formula": "Y = NOT X", "batches": [1,2,4,8,16,32,64], "precision": "位精确(bool)", "status": "COMPLETE"},
    "or":       {"shape": "[B,12,256,256] (bool)",    "dtype": "Bool", "formula": "Y = X1 OR X2", "batches": [1,2,4,8,16,32,64], "precision": "位精确(bool)", "status": "COMPLETE_WITH_LIMITATION"},
    "where":    {"shape": "[B,12,256,256]",           "dtype": "FP16+uint8", "formula": "Y = condition ? X1 : X2", "batches": [1,2,4,8,16,32,64], "precision": "位精确", "status": "COMPLETE_WITH_LIMITATION"},
    "expand":   {"shape": "X[B,256,1]→Y[B,256,384]",  "dtype": "FP16", "formula": "Y[b,i,j] = X[b,i,0]", "batches": [1,2,4,8,16,32,64], "precision": "位精确", "status": "COMPLETE_WITH_LIMITATION"},
    "transpose": {"shape": "X[B,256,384]→Y[B,384,256]", "dtype": "FP16", "formula": "Y[b,j,i] = X[b,i,j]", "batches": [1,2,4,8,16,32,64], "precision": "位精确", "status": "COMPLETE_WITH_LIMITATION"},
    "reduce_sum": {"shape": "X[B,256,384]→Y[B,256]",  "dtype": "FP16", "formula": "Y[b,i] = sum_j X[b,i,j]", "batches": [1,2,4,8,16,32,64], "precision": "atol=0.01, rtol=0.01", "status": "COMPLETE_WITH_LIMITATION"},
    "matmul":   {"shape": "A[B,12,256,256],B[B,12,256,32]", "dtype": "FP16", "formula": "Y = A @ B", "batches": [1,2,4,8,16,32,64], "precision": "atol=0.01, rtol=0.01", "status": "COMPLETE"},
}

def load_parsed(op, impl, batch):
    path = os.path.join(OPERATORS_DIR, op, "reports", "parsed", f"{impl}_b{batch}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_all_parsed(op):
    """Load all available parsed data for an operator."""
    data = {}
    for impl in ["torch", "ascendc", "pypto"]:
        impl_data = {}
        for b in OPERATOR_META[op]["batches"]:
            p = load_parsed(op, impl, b)
            if p and "error" not in p:
                impl_data[b] = p
        if impl_data:
            data[impl] = impl_data
    return data

def update_final_csv(op):
    """Update the final_comparison.csv (or provisional_comparison.csv) for an operator."""
    parsed_data = load_all_parsed(op)
    meta = OPERATOR_META[op]
    batches = meta["batches"]

    # Determine if this is provisional or final
    is_provisional = op in ["expand", "transpose", "reduce_sum"]
    csv_name = "provisional_comparison.csv" if is_provisional else "final_comparison.csv"
    csv_dir = os.path.join(OPERATORS_DIR, op, "reports", "final")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, csv_name)

    # Build rows
    rows = []
    for b in batches:
        row = {"batch": b}
        for impl in ["torch", "ascendc", "pypto"]:
            if impl in parsed_data and b in parsed_data[impl]:
                p = parsed_data[impl][b]
                row[f"{impl}_primary_us"] = p["primary_compute_kernel_us"]
                row[f"{impl}_kernels_per_call"] = p["kernels_per_logical_call"]
                row[f"{impl}_kernel_type"] = p["primary_kernel_type"]
                row[f"{impl}_all_device_us"] = p["all_device_kernels_us_per_call"]
                row[f"{impl}_kernel_name"] = p["primary_kernel_name"]
            else:
                row[f"{impl}_primary_us"] = "N/A"
                row[f"{impl}_kernels_per_call"] = "N/A"
                row[f"{impl}_kernel_type"] = "N/A"
                row[f"{impl}_all_device_us"] = "N/A"
                row[f"{impl}_kernel_name"] = "N/A"
        rows.append(row)

    # Write CSV
    fieldnames = ["batch",
                  "torch_primary_us", "torch_kernels_per_call", "torch_kernel_type", "torch_all_device_us", "torch_kernel_name",
                  "ascendc_primary_us", "ascendc_kernels_per_call", "ascendc_kernel_type", "ascendc_all_device_us", "ascendc_kernel_name",
                  "pypto_primary_us", "pypto_kernels_per_call", "pypto_kernel_type", "pypto_all_device_us", "pypto_kernel_name"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Updated {csv_path} ({len(rows)} batches)")
    return parsed_data

def update_performance_matrix(all_data):
    """Update the project-level performance matrix CSV."""
    path = os.path.join(PROJECT_ROOT, "reports", "release", "performance_matrix.csv")
    ops_order = ["relu", "mul", "add", "div", "expand", "transpose", "reduce_sum", "matmul", "equal", "not", "or", "where"]
    
    rows = []
    for op in ops_order:
        meta = OPERATOR_META[op]
        row = {"Operator": op}
        for impl, metric in [("Torch", "torch"), ("AscendC", "ascendc"), ("PyPTO", "pypto")]:
            for batch_label, batch_val in [("B1", 1), ("B32", 32)]:
                key = f"{impl}_{batch_label}_us"
                if metric in all_data.get(op, {}) and batch_val in all_data[op].get(metric, {}):
                    row[key] = all_data[op][metric][batch_val]["primary_compute_kernel_us"]
                else:
                    row[key] = "N/A"
        row["Profiler_Method"] = "msprof"
        row["Metric"] = "primary_compute_kernel_us"
        rows.append(row)

    fieldnames = ["Operator", "Torch_B1_us", "Torch_B32_us", "AscendC_B1_us", "AscendC_B32_us",
                  "PyPTO_B1_us", "PyPTO_B32_us", "Profiler_Method", "Metric"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Updated {path} ({len(rows)} operators)")

BUILTIN_LIMITATIONS = {
    "relu": [],
    "mul": [],
    "add": [{"route": "pypto", "severity": "P2", "desc": "3次二元加法→9内核事件/调用, ~46x总延迟"}],
    "div": [{"route": "pypto", "severity": "P2", "desc": "广播触发CompileFunction失败"}],
    "equal": [{"route": "pypto", "severity": "P2", "desc": "RC2解封但无msprof数据"}],
    "not": [],
    "or": [{"route": "pypto", "severity": "P1", "desc": "无logical_or API, 用bitwise_or替代"}],
    "where": [{"route": "pypto", "severity": "P2", "desc": "RC2解封但无msprof数据"}],
    "expand": [{"route": "pypto", "severity": "P2", "desc": "RC3用torch.expand().clone() workaround, 33600x提升"}],
    "transpose": [{"route": "pypto", "severity": "P2", "desc": "CompileFunction tile_shape失效"}],
    "reduce_sum": [
        {"route": "ascendc_fp16", "severity": "P1", "desc": "FP16累积384元素归约精度不足(21/70)"},
        {"route": "pypto", "severity": "P2", "desc": "无独立msprof数据"}
    ],
    "matmul": [
        {"route": "ascendc", "severity": "P3", "desc": "N=32限制Cube利用率~48%; B≥16时Torch更快"},
        {"route": "pypto", "severity": "P2", "desc": "自动tiling完全损坏(FC4000)"}
    ],
}

def build_dashboard(all_data):
    """Build Chinese dashboard.json from parsed data."""
    dashboard = {
        "version": "1.5-zh",
        "mode": "release",
        "release_version": "1.5-zh",
        "generated_at": "2026-07-20T12:00:00Z",
        "environment": {
            "platform": "Ascend 910B (dav-2201)",
            "aicore_count": 20,
            "cann_version": "9.0.0",
            "python": "3.11",
            "profiler": "msprof with --ascendcl=on --ai-core=on --task-time=l0",
            "warmup": 200,
            "profiled_loops": 100,
            "repeat": 5,
            "primary_metric": "primary_compute_kernel_us"
        },
        "operator_count": 12,
        "status_summary": {
            "COMPLETE": sum(1 for op in OPERATOR_META if OPERATOR_META[op]["status"] == "COMPLETE"),
            "COMPLETE_WITH_LIMITATION": sum(1 for op in OPERATOR_META if OPERATOR_META[op]["status"] == "COMPLETE_WITH_LIMITATION")
        },
        "status_summary_zh": {
            "完全完成": sum(1 for op in OPERATOR_META if OPERATOR_META[op]["status"] == "COMPLETE"),
            "有限完成": sum(1 for op in OPERATOR_META if OPERATOR_META[op]["status"] == "COMPLETE_WITH_LIMITATION")
        },
        "operators": {},
        "known_limitations": [],
        "pypto_limits": [
            "自动tiling系统不可靠(matmul完全损坏,div/transpose需手动覆盖)",
            "dtype推断不一致(equal误判输出类型)",
            "广播支持有限(div触发参数错误)",
            "AICPU调度开销~50-140µs/调用",
            "MIX_AIC内核比纯AIVEC重",
            "算子链分解导致多内核事件(add 4→9内核)"
        ],
        "ascendc_summary": {
            "TRUE_CUBE": ["matmul"],
            "TRUE_DEVICE": ["relu", "mul", "add", "div", "equal", "not", "or", "where", "expand", "transpose", "reduce_sum"],
            "post_rc3_changes": [
                "matmul: 多核批调度(1 kernel/call, 257x加速)",
                "reduce_sum: 新增FP32累积内核(精度3.5x提升)"
            ]
        }
    }

    for op in ["relu", "mul", "add", "div", "equal", "not", "or", "where", "expand", "transpose", "reduce_sum", "matmul"]:
        meta = OPERATOR_META[op]
        pd = all_data.get(op, {})
        
        op_entry = {
            "status": meta["status"],
            "status_zh": "完全完成" if meta["status"] == "COMPLETE" else "有限完成",
            "formula": meta["formula"],
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "batches": meta["batches"],
            "precision": meta["precision"],
            "cube_class": op == "matmul",
            "profiler": {},
            "batch_scaling": {},
            "limitations": BUILTIN_LIMITATIONS.get(op, []),
        }

        for impl, impl_label in [("torch", "torch"), ("ascendc", "ascendc"), ("pypto", "pypto")]:
            if impl in pd:
                bs = {}
                for b in meta["batches"]:
                    if b in pd[impl]:
                        bs[f"b{b}_us"] = pd[impl][b]["primary_compute_kernel_us"]
                if bs:
                    op_entry["batch_scaling"][impl_label] = bs
                
                # Profiler summary: B1 and B32
                prof = {"method": "msprof", "metric": "primary_compute_kernel_us"}
                for b in [1, 32]:
                    if b in pd.get(impl, {}):
                        p = pd[impl][b]
                        prof[f"b{b}_us"] = p["primary_compute_kernel_us"]
                        prof["kernel_type"] = p["primary_kernel_type"]
                        prof["kernel_name"] = p["primary_kernel_name"]
                        prof["kernels_per_call"] = p["kernels_per_logical_call"]
                if prof.get("b1_us") or prof.get("b32_us"):
                    op_entry["profiler"][impl_label] = prof

        dashboard["operators"][op] = op_entry
        
        # Collect limitations
        for lim in BUILTIN_LIMITATIONS.get(op, []):
            dashboard["known_limitations"].append({
                "operator": op,
                "route": lim["route"],
                "severity": lim["severity"],
                "description": lim["desc"]
            })

    path = os.path.join(PROJECT_ROOT, "dashboard", "dashboard.json")
    with open(path, "w") as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)
    print(f"  Updated {path}")

def main():
    all_data = {}
    for op in OPERATOR_META:
        print(f"Processing {op}...")
        pd = update_final_csv(op)
        all_data[op] = pd

    update_performance_matrix(all_data)
    build_dashboard(all_data)

    # Summary
    print("\n=== Summary ===")
    for op in all_data:
        ops = all_data[op]
        impls_found = list(ops.keys())
        batches_found = {}
        for impl in ops:
            batches_found[impl] = sorted(ops[impl].keys())
        print(f"  {op}: {impls_found} -> {batches_found}")

if __name__ == "__main__":
    main()
