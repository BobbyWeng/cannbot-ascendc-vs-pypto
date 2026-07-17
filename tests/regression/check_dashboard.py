#!/usr/bin/env python3
"""Validate dashboard.json completeness and consistency."""
import json
import sys
import os

EXPECTED_OPERATORS = [
    "relu", "mul", "add", "div", "equal", "not",
    "or", "where", "expand", "transpose", "reduce_sum", "matmul",
]
VALID_STATUSES = {"COMPLETE", "COMPLETE_WITH_LIMITATION", "BLOCKED", "INCOMPLETE"}
PROFILER_TOOLS = {"msprof", "torch.npu.Event", "NONE"}
REQUIRED_PROFILER_KEYS = {"torch", "ascendc", "pypto"}


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
            warnings.append(
                f"{op_name}: unexpected status '{status}'"
            )

        profiler = op_data.get("profiler", {})
        if not isinstance(profiler, dict):
            warnings.append(f"{op_name}: profiler is not a dict")
        else:
            for route_key in REQUIRED_PROFILER_KEYS:
                if route_key not in profiler:
                    warnings.append(
                        f"{op_name}/profiler: missing route '{route_key}'"
                    )
                    continue
                val = profiler[route_key]
                if val == "N/A" and op_name == "reduce_sum" and route_key == "pypto":
                    continue
                if val == "N/A":
                    warnings.append(
                        f"{op_name}/profiler/{route_key}: value is N/A"
                    )

        batches = op_data.get("batches", [])
        if not batches:
            warnings.append(f"{op_name}: no batches defined")

        correctness = op_data.get("correctness", {})
        if not correctness:
            warnings.append(f"{op_name}: missing correctness section")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "file": filepath,
        "operator_count": operator_count,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate dashboard.json")
    parser.add_argument("file", help="Path to dashboard.json")
    parser.add_argument("--expected-count", type=int, default=12,
                        help="Expected operator count")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

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

    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
