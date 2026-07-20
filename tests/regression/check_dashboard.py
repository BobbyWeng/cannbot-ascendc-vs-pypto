#!/usr/bin/env python3
"""Validate dashboard.json completeness, validation, ranking, and index.html embedding."""
import json
import sys
import os
import re

EXPECTED_OPERATORS = [
    "relu", "mul", "add", "div", "equal", "not",
    "or", "where", "expand", "transpose", "reduce_sum", "matmul",
]
VALID_STATUSES = {"COMPLETE", "COMPLETE_WITH_LIMITATION", "BLOCKED", "INCOMPLETE"}
RELEASE_SHA256 = "b235be9f2fb9261fefca1bb903a8db8f1ea6591f5b892e71ce87421232eb464a"
MUST_ALL_PASS = ["relu", "mul", "add", "div", "equal", "not", "or", "where", "expand", "transpose", "matmul"]


def _check_source(data):
    errors = []
    src = data.get("source", {})
    if not src:
        return ["Missing source section"]
    if src.get("release_sha256") != RELEASE_SHA256:
        errors.append(f"source.release_sha256={src.get('release_sha256')}, expected {RELEASE_SHA256}")
    if src.get("performance_matrix_used") is not False:
        errors.append("source.performance_matrix_used should be false")
    rf = src.get("release_file", "")
    if not rf or rf.startswith("/"):
        errors.append(f"source.release_file invalid: {rf}")
    if not src.get("data_priority"):
        errors.append("source.data_priority missing")
    return errors


def _check_matmul(operators):
    errors = []
    mm = operators.get("matmul", {})
    if not mm:
        return ["matmul not found"]
    checks = {
        "torch": [("b1_us", 12.2), ("b32_us", 63.3)],
        "ascendc": [("b1_us", 6.4), ("b32_us", 37.0), ("kernels_per_call", 1),
                    ("blockDim_b1", 12), ("blockDim_b32", 20),
                    ("b1_TFLOPS", 7.86), ("b32_TFLOPS", 43.53)],
    }
    for route_key, fields in checks.items():
        p = mm.get("profiler", {}).get(route_key, {})
        if not p:
            errors.append(f"matmul/{route_key}: profiler missing")
            continue
        for key, expected in fields:
            actual = p.get(key)
            if actual is None:
                errors.append(f"matmul/{route_key}/{key}: missing (expected {expected})")
            elif abs(float(actual) - float(expected)) > 0.001:
                errors.append(f"matmul/{route_key}/{key}: {actual} != {expected}")
        val = p.get("validation", {})
        if not val.get("rank_eligible"):
            errors.append(f"matmul/{route_key}: should be rankable but isn't ({val.get('status')})")
        if p.get("source") != "current_release.json":
            errors.append(f"matmul/{route_key}: source not current_release")
        if p.get("source_kind") != "published":
            errors.append(f"matmul/{route_key}: source_kind not published")

    # Ranking: should be RANKED with ascendc winner
    ranking = mm.get("ranking", {})
    if ranking.get("status") != "RANKED":
        errors.append(f"matmul ranking status={ranking.get('status')}, expected RANKED")
    if ranking.get("winner") != "ascendc":
        errors.append(f"matmul ranking winner={ranking.get('winner')}, expected ascendc")
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

    # FP16 legacy must NOT be rankable
    rs_prof = rs.get("profiler", {}).get("ascendc", {})
    val = rs_prof.get("validation", {})
    if val.get("rank_eligible") is not False:
        errors.append("reduce_sum/ascendc: FP16 legacy should NOT be rankable")
    if rs_prof.get("route_variant") != "ascendc_fp16_legacy":
        errors.append("reduce_sum/ascendc: missing route_variant")
    if rs_prof.get("comparable") is not False:
        errors.append("reduce_sum/ascendc: comparable should be False")
    if not rs_prof.get("comparison_note"):
        errors.append("reduce_sum/ascendc: missing comparison_note")

    # Torch correctness is PARTIAL -> not rankable
    torch_prof = rs.get("profiler", {}).get("torch", {})
    tval = torch_prof.get("validation", {})
    if tval.get("rank_eligible"):
        errors.append("reduce_sum/torch: correctness=62/70 should NOT be rankable")

    # Ranking
    ranking = rs.get("ranking", {})
    if ranking.get("winner") is not None:
        errors.append("reduce_sum: should have no winner (insufficient verified routes)")

    # correctness_notes
    notes = rs.get("correctness_notes", {})
    if "FP16" not in notes.get("profiler_note", ""):
        errors.append("reduce_sum: profiler_note missing FP16 annotation")
    # ascendc_fp16 must say PARTIAL not PASS
    fp16_note = notes.get("ascendc_fp16", "")
    if "PARTIAL" not in fp16_note:
        errors.append(f"reduce_sum/ascendc_fp16: should say PARTIAL, got '{fp16_note}'")
    if "PASS" in fp16_note:
        errors.append(f"reduce_sum/ascendc_fp16: must not say PASS, got '{fp16_note}'")
    return errors


def _check_correctness(operators):
    errors = []
    for op_name in EXPECTED_OPERATORS:
        corr = operators.get(op_name, {}).get("correctness", {})
        for rk in ["torch", "ascendc", "pypto"]:
            if rk not in corr:
                errors.append(f"{op_name}/correctness: missing route '{rk}'")
            elif not corr[rk] or corr[rk] == "N/A":
                errors.append(f"{op_name}/correctness/{rk}: is N/A")

    for op_name in MUST_ALL_PASS:
        corr = operators.get(op_name, {}).get("correctness", {})
        for rk in ["torch", "ascendc", "pypto"]:
            val = str(corr.get(rk, "")).strip()
            if not val.startswith("PASS"):
                errors.append(f"{op_name}/correctness/{rk}: '{val}' does not start with PASS")
    return errors


def _check_profiler_validation(operators):
    errors = []
    for op_name in EXPECTED_OPERATORS:
        prof = operators.get(op_name, {}).get("profiler", {})
        for rk in ["torch", "ascendc", "pypto"]:
            p = prof.get(rk, {})
            if not p or p.get("b1_us") is None:
                continue
            val = p.get("validation", {})
            if not val:
                errors.append(f"{op_name}/profiler/{rk}: missing validation")
                continue
            # Every profiler with b1_us should have a validation status
            if not val.get("status"):
                errors.append(f"{op_name}/profiler/{rk}: validation status missing")
            if val.get("rank_eligible") is None:
                errors.append(f"{op_name}/profiler/{rk}: rank_eligible missing")

    # Validation summary must exist
    return errors


def _check_validation_summary(data, operators):
    errors = []
    vs = data.get("validation_summary", {})
    if not vs:
        return ["missing validation_summary"]
    required_keys = ["verified_rankable", "verified_not_comparable", "unmanifested",
                     "hash_mismatch", "correctness_not_pass", "variant_mismatch",
                     "missing_profiler", "total_operators"]
    for k in required_keys:
        if k not in vs:
            errors.append(f"validation_summary missing '{k}'")
    if vs.get("total_operators") != 12:
        errors.append(f"validation_summary.total_operators={vs.get('total_operators')}")

    # insufficient_routes must match actual count of INSUFFICIENT rankings
    ir_summary = vs.get("insufficient_routes", 0)
    ir_actual = sum(
        1 for op in operators.values()
        if op.get("ranking", {}).get("status") == "INSUFFICIENT_VERIFIED_ROUTES"
    )
    if ir_summary != ir_actual:
        errors.append(f"validation_summary.insufficient_routes={ir_summary} but actual count={ir_actual}")

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


def _check_ranking_edge_cases(operators):
    """Regression tests for ranking behavior."""
    errors = []

    # All operators with SHA256-verified profilers should produce ranking
    # (manifests exist for their parsed files)
    for op_name in EXPECTED_OPERATORS:
        ranking = operators.get(op_name, {}).get("ranking", {})
        prof = operators.get(op_name, {}).get("profiler", {})
        rankable_count = sum(1 for p in prof.values()
                             if isinstance(p, dict) and p.get("validation", {}).get("rank_eligible"))
        if rankable_count >= 2:
            if ranking.get("status") != "RANKED":
                errors.append(f"{op_name}: {rankable_count} rankable routes but ranking={ranking.get('status')}")
            if ranking.get("winner") is None:
                errors.append(f"{op_name}: {rankable_count} rankable routes but no winner")
        elif rankable_count == 1:
            if ranking.get("status") != "INSUFFICIENT_VERIFIED_ROUTES":
                errors.append(f"{op_name}: 1 rankable route, expected INSUFFICIENT but got {ranking.get('status')}")

    # MatMul must be RANKED
    mm_ranking = operators.get("matmul", {}).get("ranking", {})
    if mm_ranking.get("status") != "RANKED":
        errors.append(f"matmul: ranking should be RANKED but got {mm_ranking.get('status')}")

    # reduce_sum: all non-rankable (torch=62/70, ascendc=FP16 legacy)
    rs_prof = operators.get("reduce_sum", {}).get("profiler", {})
    for rk in ["torch", "ascendc"]:
        p = rs_prof.get(rk, {})
        if isinstance(p, dict) and p.get("validation", {}).get("rank_eligible"):
            errors.append(f"reduce_sum/{rk}: should NOT be rankable")

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

    errors.extend(_check_schema(data))
    errors.extend(_check_source(data))
    errors.extend(_check_correctness(operators))
    errors.extend(_check_matmul(operators))
    errors.extend(_check_reduce_sum(operators))
    errors.extend(_check_profiler_validation(operators))
    errors.extend(_check_validation_summary(data, operators))
    errors.extend(_check_ranking_edge_cases(operators))

    for op_name in EXPECTED_OPERATORS:
        if op_name not in operators:
            continue
        status = operators[op_name].get("status", "")
        if status not in VALID_STATUSES:
            warnings.append(f"{op_name}: unexpected status '{status}'")

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
        errors.append("Missing embedded script")
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
    if 'JSON.parse(embedded.textContent)' not in html:
        errors.append("init() does not read embedded data before fetch")
    if "fetch('./dashboard.json')" not in html:
        errors.append("init() missing fetch fallback")
    status = "PASS" if not errors else "FAIL"
    return {"status": status, "errors": errors, "file": index_html_path}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default=None, help="Path to dashboard.json")
    parser.add_argument("--expected-count", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--index-html", default=None)
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
