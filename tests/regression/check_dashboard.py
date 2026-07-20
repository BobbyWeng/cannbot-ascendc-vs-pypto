#!/usr/bin/env python3
"""Validate dashboard.json completeness and index.html embedding consistency."""
import json
import sys
import os
import re

EXPECTED_OPERATORS = [
    "relu", "mul", "add", "div", "equal", "not",
    "or", "where", "expand", "transpose", "reduce_sum", "matmul",
]
VALID_STATUSES = {"COMPLETE", "COMPLETE_WITH_LIMITATION", "BLOCKED", "INCOMPLETE"}

# Per-operator correctness expectations
# "PASS" = at least one route shows PASS; "PARTIAL" = mixed expected; "N/A" = all N/A acceptable
CORRECTNESS_EXPECTED = {
    "relu":     {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "mul":      {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "add":      {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "div":      {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "equal":    {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "not":      {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "or":       {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "where":    {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "expand":   {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "transpose":{"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
    "reduce_sum": {"torch": "PARTIAL", "ascendc": "PASS", "pypto": "PASS"},
    "matmul":   {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
}

# Operators where torch and ascendc MUST have profiler b1_us data
PROFILER_REQUIRED_B1 = {
    "torch": ["relu", "mul", "add", "div", "equal", "not", "or", "where", "expand", "transpose", "reduce_sum", "matmul"],
    "ascendc": ["relu", "mul", "add", "div", "equal", "not", "or", "where", "expand", "transpose", "reduce_sum", "matmul"],
}

# Operators that must NOT show all-N/A correctness
MUST_SHOW_ROUTE_CORRECTNESS = ["relu", "mul", "add", "not", "matmul"]


def check_dashboard(filepath: str, expected_count: int = 12):
    errors = []
    warnings = []

    if not os.path.exists(filepath):
        return {"status": "FAIL", "errors": [f"File not found: {filepath}"]}

    with open(filepath) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return {"status": "FAIL", "errors": [f"Invalid JSON: {e}"]}

    operator_count = data.get("operator_count", 0)
    if operator_count != expected_count:
        errors.append(
            f"operator_count={operator_count}, expected {expected_count}"
        )

    operators = data.get("operators", {})
    if len(operators) != expected_count:
        errors.append(
            f"operators dict has {len(operators)} entries, expected {expected_count}"
        )

    for op_name in EXPECTED_OPERATORS:
        if op_name not in operators:
            errors.append(f"Missing operator '{op_name}' in dashboard")
            continue
        op_data = operators[op_name]
        status = op_data.get("status", "")
        if status not in VALID_STATUSES:
            warnings.append(f"{op_name}: unexpected status '{status}'")

        # correctness route-level checks
        corr = op_data.get("correctness", {})
        if not isinstance(corr, dict):
            errors.append(f"{op_name}: correctness is not a dict")
        else:
            expected_corr = CORRECTNESS_EXPECTED.get(op_name, {})
            for route_key, expected_val in expected_corr.items():
                actual = corr.get(route_key, "N/A")
                if actual == "N/A":
                    if expected_val != "PASS":
                        continue  # PARTIAL case: N/A is OK
                    warnings.append(f"{op_name}/correctness/{route_key}: N/A (expected {expected_val})")

        # profiler checks
        profiler = op_data.get("profiler", {})
        if not isinstance(profiler, dict):
            errors.append(f"{op_name}: profiler is not a dict")
        else:
            for route_key in ["torch", "ascendc", "pypto"]:
                p = profiler.get(route_key, {})
                if not isinstance(p, dict):
                    continue
                b1 = p.get("b1_us")
                if route_key in PROFILER_REQUIRED_B1 and op_name in PROFILER_REQUIRED_B1[route_key]:
                    if b1 is None:
                        errors.append(f"{op_name}/profiler/{route_key}: b1_us is missing (required)")
                    elif not isinstance(b1, (int, float)):
                        warnings.append(f"{op_name}/profiler/{route_key}: b1_us={b1} is not numeric")

        # batches
        batches = op_data.get("batches", [])
        if not batches:
            warnings.append(f"{op_name}: no batches defined")

        # correctness section must exist
        if not corr:
            errors.append(f"{op_name}: missing correctness data")

        # Key operators must NOT have all-N/A correctness
        if op_name in MUST_SHOW_ROUTE_CORRECTNESS:
            all_na = all(v == "N/A" for v in corr.values()) if corr else True
            if all_na:
                errors.append(f"{op_name}: all correctness routes are N/A (must show route-level data)")

        # batch_scaling should exist for torch and ascendc
        bs = op_data.get("batch_scaling", {})
        for route_key in ["torch", "ascendc"]:
            if route_key in PROFILER_REQUIRED_B1 and op_name in PROFILER_REQUIRED_B1[route_key]:
                if route_key not in bs:
                    warnings.append(f"{op_name}/batch_scaling: missing '{route_key}'")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "file": filepath,
        "operator_count": operator_count,
    }


def check_index_html(index_html_path: str, dashboard_json_path: str):
    """Verify index.html embeds dashboard data correctly."""
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
    parser.add_argument("--expected-count", type=int, default=12,
                        help="Expected operator count")
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
            if "operator_count" in result:
                print(f"  Operators: {result['operator_count']}")
            for e in result.get("errors", []):
                print(f"  ERROR: {e}")
            for w in result.get("warnings", []):
                print(f"  WARN: {w}")
        if result["status"] == "FAIL":
            all_pass = False

    if args.index_html:
        dj_path = args.file or (os.path.join(os.path.dirname(args.index_html), "dashboard.json"))
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
