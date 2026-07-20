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

# MatMul post-RC3 exact values (from current_release.json)
MATMUL = {
    "torch": {"b1_us": 12.2, "b32_us": 63.3},
    "ascendc": {"b1_us": 6.4, "b32_us": 37.0, "kernels_per_call": 1,
                "blockDim_b1": 12, "blockDim_b32": 20,
                "b1_TFLOPS": 7.86, "b32_TFLOPS": 43.53},
}

RELEASE_SHA256 = "b235be9f2fb9261fefca1bb903a8db8f1ea6591f5b892e71ce87421232eb464a"

# Operators that MUST have all 3 correctness routes and each starts with PASS
MUST_ALL_PASS = ["relu", "mul", "add", "div", "equal", "not", "or", "where", "expand", "transpose", "matmul"]


def _check_source(data):
    errors = []
    src = data.get("source", {})
    if not src:
        return ["Missing source section"]

    # release_sha256
    actual_sha = src.get("release_sha256", "")
    if actual_sha != RELEASE_SHA256:
        errors.append(f"source.release_sha256={actual_sha}, expected {RELEASE_SHA256}")

    # performance_matrix not used
    if src.get("performance_matrix_used") is not False:
        errors.append("source.performance_matrix_used should be false")

    # release_file is repo-relative, not absolute
    rf = src.get("release_file", "")
    if not rf:
        errors.append("source.release_file is empty")
    elif rf.startswith("/"):
        errors.append(f"source.release_file is absolute path: {rf}")

    # data_priority exists
    if not src.get("data_priority"):
        errors.append("source.data_priority missing")

    return errors


def _check_matmul(operators):
    errors = []
    mm = operators.get("matmul", {})
    if not mm:
        return ["matmul not found"]

    for route_key, checks in MATMUL.items():
        p = mm.get("profiler", {}).get(route_key, {})
        if not p:
            errors.append(f"matmul/{route_key}: profiler missing")
            continue
        # Exact value checks
        for key, expected in checks.items():
            actual = p.get(key)
            if actual is None:
                errors.append(f"matmul/{route_key}/{key}: missing (expected {expected})")
            elif abs(float(actual) - float(expected)) > 0.001:
                errors.append(f"matmul/{route_key}/{key}: {actual} != {expected}")
        # Provenance
        for prov_key in ["source", "source_kind", "integrity", "comparable"]:
            if prov_key not in p:
                errors.append(f"matmul/{route_key}: missing provenance field '{prov_key}'")
        if p.get("source") != "current_release.json":
            errors.append(f"matmul/{route_key}: source not current_release.json")
        if p.get("source_kind") != "published":
            errors.append(f"matmul/{route_key}: source_kind not published")
        if not str(p.get("integrity", "")).startswith("sha256:"):
            errors.append(f"matmul/{route_key}: integrity missing sha256 prefix")
        if p.get("comparable") is not True:
            errors.append(f"matmul/{route_key}: comparable not True")

    return errors


def _check_reduce_sum(operators):
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

    # FP16/FP32 separation
    rs_prof = rs.get("profiler", {}).get("ascendc", {})
    if rs_prof.get("b1_us"):
        if rs_prof.get("route_variant") != "ascendc_fp16_legacy":
            errors.append(f"reduce_sum/ascendc: missing route_variant='ascendc_fp16_legacy'")
        if rs_prof.get("comparable") is not False:
            errors.append(f"reduce_sum/ascendc: comparable should be False (FP16 legacy)")
        if not rs_prof.get("comparison_note"):
            errors.append(f"reduce_sum/ascendc: missing comparison_note")

    # correctness_notes
    notes = rs.get("correctness_notes", {})
    if "FP16" not in notes.get("profiler_note", ""):
        errors.append(f"reduce_sum: profiler_note missing FP16 annotation")

    return errors


def _check_correctness(operators):
    errors = []
    for op_name in EXPECTED_OPERATORS:
        op_data = operators.get(op_name, {})
        corr = op_data.get("correctness", {})
        for rk in ["torch", "ascendc", "pypto"]:
            if rk not in corr:
                errors.append(f"{op_name}/correctness: missing route '{rk}'")
                continue
            val = corr[rk]
            if not val or val == "N/A":
                errors.append(f"{op_name}/correctness/{rk}: is N/A (must be parsed from coverage)")

    # Full PASS check for MUST_ALL_PASS operators
    for op_name in MUST_ALL_PASS:
        corr = operators.get(op_name, {}).get("correctness", {})
        for rk in ["torch", "ascendc", "pypto"]:
            val = str(corr.get(rk, ""))
            if not val.startswith("PASS"):
                errors.append(f"{op_name}/correctness/{rk}: '{val}' does not start with PASS (expected PASS)")
    return errors


def _check_profiler_provenance(operators):
    errors = []
    for op_name in EXPECTED_OPERATORS:
        prof = operators.get(op_name, {}).get("profiler", {})
        for rk in ["torch", "ascendc", "pypto"]:
            p = prof.get(rk, {})
            if not p or p.get("b1_us") is None:
                continue
            # Every profiler entry with b1_us must have full provenance
            for prov_key in ["source", "source_kind", "sha256", "integrity", "comparable"]:
                if prov_key not in p:
                    errors.append(f"{op_name}/profiler/{rk}: missing provenance '{prov_key}'")
            # sha256 must not be None/empty
            sha_val = p.get("sha256")
            if sha_val is None or (isinstance(sha_val, str) and sha_val == ""):
                errors.append(f"{op_name}/profiler/{rk}: sha256 is empty")
    return errors


def _check_schema(data):
    errors = []
    if data.get("schema_version") != "1.0":
        errors.append(f"schema_version={data.get('schema_version')}, expected 1.0")
    if data.get("release_version") != "1.4-post-rc3":
        errors.append(f"release_version={data.get('release_version')}, expected 1.4-post-rc3")
    if data.get("generated_at") != "2026-07-17T15:30:00Z":
        errors.append(f"generated_at={data.get('generated_at')}, expected 2026-07-17T15:30:00Z")
    return errors


def check_dashboard(filepath: str, expected_count: int = 12):
    errors = []
    warnings = []

    if not os.path.exists(filepath):
        return {"status": "FAIL", "errors": [f"File not found: {filepath}"]}

    with open(filepath) as f:
        try:
            data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return {"status": "FAIL", "errors": [f"Invalid JSON: {e}"]}

    operator_count = data.get("operator_count", 0)
    if operator_count != expected_count:
        errors.append(f"operator_count={operator_count}, expected {expected_count}")
    operators = data.get("operators", {})
    if len(operators) != expected_count:
        errors.append(f"operators dict has {len(operators)} entries, expected {expected_count}")

    # Run all sub-checks
    errors.extend(_check_schema(data))
    errors.extend(_check_source(data))
    errors.extend(_check_correctness(operators))
    errors.extend(_check_matmul(operators))
    errors.extend(_check_reduce_sum(operators))
    errors.extend(_check_profiler_provenance(operators))

    for op_name in EXPECTED_OPERATORS:
        if op_name not in operators:
            continue
        op_data = operators[op_name]
        status = op_data.get("status", "")
        if status not in VALID_STATUSES:
            warnings.append(f"{op_name}: unexpected status '{status}'")
        batches = op_data.get("batches", [])
        if not batches:
            warnings.append(f"{op_name}: no batches defined")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "file": filepath,
        "operator_count": operator_count,
        "source_sha256_match": (data.get("source", {}).get("release_sha256") == RELEASE_SHA256),
        "schema_version": data.get("schema_version"),
    }


def check_index_html(index_html_path: str, dashboard_json_path: str):
    errors = []
    if not os.path.exists(index_html_path):
        return {"status": "FAIL", "errors": [f"File not found: {index_html_path}"]}

    html = open(index_html_path).read()

    m = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>', html, re.DOTALL)
    if not m:
        errors.append("Missing <script id='dashboard-data'>")
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
        errors.append(f"dashboard.json not found: {dashboard_json_path}")

    if 'JSON.parse(embedded.textContent)' not in html:
        errors.append("init() does not read embedded data before fetch")
    if "fetch('./dashboard.json')" not in html:
        errors.append("init() missing fetch fallback")

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
            print(f"  Schema: {result.get('schema_version', '?')}")
            print(f"  SHA256 match: {result.get('source_sha256_match', '?')}")
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
