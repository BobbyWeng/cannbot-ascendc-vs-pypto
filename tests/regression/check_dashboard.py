#!/usr/bin/env python3
"""验证 Dashboard v3 数据模型、来源完整性和关键发布数据。"""

import argparse
import json
import os
import re
import sys


EXPECTED_OPERATORS = {
    "relu", "mul", "add", "div", "equal", "not",
    "or", "where", "expand", "transpose", "reduce_sum", "matmul",
}
ROUTES = {"torch", "ascendc", "pypto"}
VALID_STATUSES = {"COMPLETE", "COMPLETE_WITH_LIMITATION", "PARTIAL", "INCOMPLETE", "BLOCKED"}
VALID_INTEGRITY = {"RELEASE_SOURCE", "VERIFIED", "UNMANIFESTED"}


def check_dashboard(filepath: str, expected_count: int = 12):
    errors = []
    warnings = []
    if not os.path.isfile(filepath):
        return {"status": "FAIL", "errors": [f"File not found: {filepath}"]}
    try:
        with open(filepath, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        return {"status": "FAIL", "errors": [f"Invalid JSON: {error}"]}

    if data.get("schema_version") != "3.0":
        errors.append(f"schema_version={data.get('schema_version')}, expected 3.0")
    if data.get("language") != "zh-CN":
        errors.append("language must be zh-CN")
    source = data.get("source", {})
    if source.get("release_file") != "reports/release/current_release.json":
        errors.append("current_release.json is not the declared release source")
    if source.get("performance_matrix_used") is not False:
        errors.append("stale performance_matrix must not be used")
    if not re.fullmatch(r"[0-9a-f]{64}", source.get("release_sha256", "")):
        errors.append("release source SHA256 is missing or invalid")

    operators = data.get("operators", {})
    if len(operators) != expected_count:
        errors.append(f"operator count={len(operators)}, expected {expected_count}")
    missing = EXPECTED_OPERATORS - set(operators)
    extra = set(operators) - EXPECTED_OPERATORS
    if missing:
        errors.append(f"missing operators: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected operators: {sorted(extra)}")

    counted = {"release_source": 0, "verified": 0, "unmanifested": 0,
               "no_kernel_evidence": 0}
    for name, operator in operators.items():
        if operator.get("status") not in VALID_STATUSES:
            errors.append(f"{name}: invalid status {operator.get('status')}")
        if not operator.get("status_zh"):
            errors.append(f"{name}: missing Chinese status")
        correctness = operator.get("correctness", {})
        if set(correctness) != ROUTES:
            errors.append(f"{name}: correctness routes must be torch/ascendc/pypto")
        for route, result in correctness.items():
            if result.get("status") not in {"PASS", "PARTIAL", "FAIL", "N/A"}:
                errors.append(f"{name}/{route}: invalid correctness status")
            if not result.get("detail"):
                errors.append(f"{name}/{route}: missing correctness evidence detail")

        performance = operator.get("performance", {})
        if performance.get("primary_metric") != "primary_compute_kernel_us":
            errors.append(f"{name}: invalid primary metric")
        for batch, route_records in performance.get("batches", {}).items():
            if not str(batch).isdigit():
                errors.append(f"{name}: invalid batch key {batch}")
            for route, record in route_records.items():
                if route not in ROUTES:
                    errors.append(f"{name}/B{batch}: unknown route {route}")
                integrity = record.get("integrity")
                if integrity not in VALID_INTEGRITY:
                    errors.append(f"{name}/B{batch}/{route}: invalid integrity {integrity}")
                if not record.get("source") or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
                    errors.append(f"{name}/B{batch}/{route}: missing provenance")
                primary = record.get("primary_us")
                if record.get("metric_status") == "AVAILABLE":
                    if primary is None or primary <= 0:
                        errors.append(f"{name}/B{batch}/{route}: non-positive available metric")
                elif record.get("metric_status") == "NO_KERNEL_EVIDENCE":
                    counted["no_kernel_evidence"] += 1
                    if primary is not None:
                        errors.append(f"{name}/B{batch}/{route}: placeholder must be N/A")
                else:
                    errors.append(f"{name}/B{batch}/{route}: invalid metric_status")
                if integrity == "RELEASE_SOURCE":
                    counted["release_source"] += 1
                elif integrity == "VERIFIED":
                    counted["verified"] += 1
                else:
                    counted["unmanifested"] += 1
                    if record.get("comparable"):
                        errors.append(f"{name}/B{batch}/{route}: unmanifested record cannot be ranked")

    evidence = data.get("summary", {}).get("evidence", {})
    for key, value in counted.items():
        if evidence.get(key) != value:
            errors.append(f"summary.evidence.{key}={evidence.get(key)}, counted {value}")
    if evidence.get("mismatch") != 0:
        errors.append(f"performance evidence has {evidence.get('mismatch')} hash mismatches")

    # Post-RC3 MatMul 必须使用 current_release 新多核数据，不能回退旧 parsed 值。
    matmul = operators.get("matmul", {}).get("performance", {}).get("batches", {})
    matmul_b1 = matmul.get("1", {}).get("ascendc", {})
    matmul_b32 = matmul.get("32", {}).get("ascendc", {})
    if matmul_b1.get("primary_us") != 6.4 or matmul_b32.get("primary_us") != 37.0:
        errors.append("matmul post-RC3 Ascend C latency must be B1=6.4us, B32=37.0us")
    if matmul_b32.get("tflops") != 43.53 or matmul_b32.get("block_dim") != 20:
        errors.append("matmul B32 TFLOPS/blockDim post-RC3 evidence is wrong")
    if any(record.get("source_kind") != "CURRENT_RELEASE"
           for records in matmul.values() for record in records.values()):
        errors.append("matmul structured routes must not mix stale parsed profiler data")

    reduce_sum = operators.get("reduce_sum", {})
    if reduce_sum.get("correctness", {}).get("ascendc", {}).get("detail") != "FP32 70/70 PASS; FP16 21/70 PARTIAL":
        errors.append("reduce_sum FP32/FP16 correctness distinction is missing")
    for records in reduce_sum.get("performance", {}).get("batches", {}).values():
        ascendc = records.get("ascendc")
        if ascendc and ascendc.get("comparable"):
            errors.append("reduce_sum legacy FP16 profiler must not rank as FP32 performance")

    if operators.get("or", {}).get("correctness", {}).get("pypto", {}).get("status") != "PASS":
        errors.append("or PyPTO correctness must parse as PASS")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "file": filepath,
        "operator_count": len(operators),
        "evidence": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Dashboard v3")
    parser.add_argument("file")
    parser.add_argument("--expected-count", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check_dashboard(args.file, args.expected_count)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[{result['status']}] {args.file}")
        print(f"  Operators: {result.get('operator_count', 0)}")
        for error in result.get("errors", []):
            print(f"  ERROR: {error}")
        for warning in result.get("warnings", []):
            print(f"  WARN: {warning}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
