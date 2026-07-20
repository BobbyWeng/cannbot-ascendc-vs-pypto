#!/usr/bin/env python3
"""Validate dashboard.json completeness and index.html embedding consistency."""
import json
import sys
import os
import re
import hashlib

EXPECTED_OPERATORS = [
    "relu", "mul", "add", "div", "equal", "not",
    "or", "where", "expand", "transpose", "reduce_sum", "matmul",
]
VALID_STATUSES = {"COMPLETE", "COMPLETE_WITH_LIMITATION", "BLOCKED", "INCOMPLETE"}

# MatMul post-RC3 exact values (from current_release.json routes.*.profiler)
MATMUL_TORCH_B1 = 12.2
MATMUL_TORCH_B32 = 63.3
MATMUL_ASCENDC_B1 = 6.4
MATMUL_ASCENDC_B32 = 37.0
MATMUL_ASCENDC_KC = 1
MATMUL_ASCENDC_BLOCKDIM_B1 = 12
MATMUL_ASCENDC_BLOCKDIM_B32 = 20
MATMUL_ASCENDC_TFLOPS_B1 = 7.86
MATMUL_ASCENDC_TFLOPS_B32 = 43.53


def _check_source_info(data):
    """Validate source tracking metadata."""
    errors = []
    src = data.get("source", {})
    if not src:
        errors.append("Missing source tracking section")
        return errors
    if src.get("release_file"):
        if not os.path.exists(src["release_file"]):
            errors.append(f"source.release_file not found: {src['release_file']}")
    if src.get("performance_matrix_used") is not False:
        errors.append("source.performance_matrix_used should be false")
    return errors


def _check_matmul(operators):
    """Validate MatMul post-RC3 exact profiler values."""
    errors = []
    mm = operators.get("matmul", {})
    if not mm:
        return ["matmul not found"]

    prof = mm.get("profiler", {})
    for route_key, checks in [
        ("torch", [("b1_us", MATMUL_TORCH_B1), ("b32_us", MATMUL_TORCH_B32)]),
        ("ascendc", [
            ("b1_us", MATMUL_ASCENDC_B1), ("b32_us", MATMUL_ASCENDC_B32),
            ("kernels_per_call", MATMUL_ASCENDC_KC),
            ("blockDim_b1", MATMUL_ASCENDC_BLOCKDIM_B1),
            ("blockDim_b32", MATMUL_ASCENDC_BLOCKDIM_B32),
            ("b1_TFLOPS", MATMUL_ASCENDC_TFLOPS_B1),
            ("b32_TFLOPS", MATMUL_ASCENDC_TFLOPS_B32),
        ]),
    ]:
        p = prof.get(route_key, {})
        src = p.get("source", "")
        if "current_release" not in src:
            errors.append(f"matmul/{route_key}: source must be from current_release (got '{src}')")
        for key, expected in checks:
            actual = p.get(key)
            if actual is None:
                errors.append(f"matmul/{route_key}/{key}: missing (expected {expected})")
            elif abs(float(actual) - float(expected)) > 0.01:
                errors.append(f"matmul/{route_key}/{key}: {actual} != expected {expected}")
    return errors


def _check_reduce_sum(operators):
    """Validate ReduceSum correctness and FP16/FP32 distinction."""
    errors = []
    rs = operators.get("reduce_sum", {})
    if not rs:
        return ["reduce_sum not found"]

    corr = rs.get("correctness", {})
    tc = corr.get("torch", "")
    ac = corr.get("ascendc", "")
    pc = corr.get("pypto", "")
    if "62" not in tc:
        errors.append(f"reduce_sum/torch: expected '62/70', got '{tc}'")
    if "FP32" not in ac or "70/70" not in ac:
        errors.append(f"reduce_sum/ascendc: expected FP32 70/70, got '{ac}'")
    if "70/70" not in pc:
        errors.append(f"reduce_sum/pypto: expected 70/70, got '{pc}'")

    notes = rs.get("correctness_notes", {})
    prof_note = notes.get("profiler_note", "")
    if "FP16" not in prof_note:
        errors.append(f"reduce_sum: profiler missing FP16 note: '{prof_note}'")
    if "FP32" not in prof_note:
        errors.append(f"reduce_sum: profiler missing FP32 note: '{prof_note}'")

    return errors


def _check_correctness(operators):
    """Validate correctness coverage parsing: key operators must not be all N/A."""
    errors = []
    for op_name in ["relu", "mul", "add", "not", "matmul"]:
        corr = operators.get(op_name, {}).get("correctness", {})
        vals = [corr.get(r, "N/A") for r in ["torch", "ascendc", "pypto"]]
        if all(v == "N/A" for v in vals):
            errors.append(f"{op_name}: all correctness routes N/A (must show parsed coverage)")
    return errors


def _check_profiler_zero(operators):
    """Any primary_compute_kernel_us of 0 must be shown as N/A."""
    errors = []
    for name, op in operators.items():
        for rk in ["torch", "ascendc", "pypto"]:
            p = op.get("profiler", {}).get(rk, {})
            for bk in ["b1_us", "b2_us", "b4_us", "b8_us", "b16_us", "b32_us", "b64_us"]:
                val = p.get(bk)
                if val is not None and val == 0:
                    errors.append(f"{name}/profiler/{rk}/{bk}: 0 value should be N/A")
    return errors


def check_dashboard(filepath: str, expected_count: int = 12):
    errors = []
    warnings = []

    if not os.path.exists(filepath):
        return {"status": "FAIL", "errors": [f"File not found: {filepath}"]}

    with open(filepath) as f:
        try:
            raw = f.read()
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            return {"status": "FAIL", "errors": [f"Invalid JSON: {e}"]}

    # Basic count checks
    operator_count = data.get("operator_count", 0)
    if operator_count != expected_count:
        errors.append(f"operator_count={operator_count}, expected {expected_count}")
    operators = data.get("operators", {})
    if len(operators) != expected_count:
        errors.append(f"operators dict has {len(operators)} entries, expected {expected_count}")

    # Release metadata
    if data.get("release_version") != "1.4-post-rc3":
        errors.append(f"release_version={data.get('release_version')}, expected 1.4-post-rc3")
    if data.get("generated_at") != "2026-07-17T15:30:00Z":
        errors.append(f"generated_at={data.get('generated_at')}, expected 2026-07-17T15:30:00Z")

    # Source validation
    errors.extend(_check_source_info(data))

    # Route-level correctness per operator
    for op_name in EXPECTED_OPERATORS:
        if op_name not in operators:
            errors.append(f"Missing operator '{op_name}'")
            continue
        op_data = operators[op_name]
        status = op_data.get("status", "")
        if status not in VALID_STATUSES:
            warnings.append(f"{op_name}: unexpected status '{status}'")

        corr = op_data.get("correctness", {})
        if not corr:
            errors.append(f"{op_name}: missing correctness section")

        profiler = op_data.get("profiler", {})
        if not isinstance(profiler, dict):
            errors.append(f"{op_name}: profiler is not a dict")

        batches = op_data.get("batches", [])
        if not batches:
            warnings.append(f"{op_name}: no batches defined")

    # Special sub-checks
    errors.extend(_check_matmul(operators))
    errors.extend(_check_reduce_sum(operators))
    errors.extend(_check_correctness(operators))
    errors.extend(_check_profiler_zero(operators))

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "file": filepath,
        "operator_count": operator_count,
    }


def check_index_html(index_html_path: str, dashboard_json_path: str):
    errors = []
    if not os.path.exists(index_html_path):
        return {"status": "FAIL", "errors": [f"File not found: {index_html_path}"]}

    html = open(index_html_path).read()

    m = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>', html, re.DOTALL)
    if not m:
        errors.append("Missing <script id='dashboard-data'> with embedded JSON")
        return {"status": "FAIL", "errors": errors}

    embedded_raw = m.group(1)
    if not embedded_raw.strip():
        errors.append("Embedded data is empty")
        return {"status": "FAIL", "errors": errors}

    try:
        embedded_data = json.loads(embedded_raw)
    except json.JSONDecodeError as e:
        errors.append(f"Embedded JSON parse error: {e}")
        return {"status": "FAIL", "errors": errors}

    if embedded_data.get("operator_count") != 12:
        errors.append(f"Embedded operator_count={embedded_data.get('operator_count')}, expected 12")
    if len(embedded_data.get("operators", {})) != 12:
        errors.append(f"Embedded operators dict length={len(embedded_data.get('operators',{}))}, expected 12")

    if os.path.exists(dashboard_json_path):
        dj = json.load(open(dashboard_json_path))
        if json.dumps(embedded_data, sort_keys=True) != json.dumps(dj, sort_keys=True):
            errors.append("Embedded data does not match dashboard.json")
    else:
        errors.append(f"dashboard.json not found for comparison: {dashboard_json_path}")

    if 'JSON.parse(embedded.textContent)' not in html:
        errors.append("init() does not read embedded data before fetch")
    if "fetch('./dashboard.json')" not in html:
        errors.append("init() missing fetch fallback for HTTP mode")

    status = "PASS" if not errors else "FAIL"
    return {"status": status, "errors": errors, "file": index_html_path}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate dashboard.json and index.html")
    parser.add_argument("file", nargs="?", default=None, help="Path to dashboard.json")
    parser.add_argument("--expected-count", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--index-html", default=None, help="Path to index.html for embedding check")
    args = parser.parse_args()

    all_pass = True

    if args.file:
        result = check_dashboard(args.file, expected_count=args.expected_count)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"[{result['status']}] {args.file}")
            print(f"  Operators: {result.get('operator_count', '?')}")
            for e in result.get("errors", []):
                print(f"  ERROR: {e}")
            for w in result.get("warnings", []):
                print(f"  WARN: {w}")
        if result["status"] == "FAIL":
            all_pass = False

    if args.index_html:
        dj_path = args.file or os.path.join(os.path.dirname(args.index_html), "dashboard.json")
        result_idx = check_index_html(args.index_html, dj_path)
        if args.json:
            print(json.dumps(result_idx, indent=2))
        else:
            print(f"[{result_idx['status']}] {args.index_html}")
            for e in result_idx.get("errors", []):
                print(f"  ERROR: {e}")
        if result_idx["status"] == "FAIL":
            all_pass = False

    if not args.file and not args.index_html:
        parser.print_help()
        sys.exit(1)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
