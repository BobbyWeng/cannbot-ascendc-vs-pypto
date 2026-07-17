#!/usr/bin/env python3
"""Validate final_comparison.json for a single operator."""
import json
import sys
import os

REQUIRED_ROUTES = {"torch", "ascendc", "pypto"}
REQUIRED_BATCH_FIELDS = [
    "primary_compute_kernel_us",
    "kernels_per_call",
]
OPTIONAL_BATCH_FIELDS = [
    "all_device_kernels_us_per_call",
    "all_device_kernels_us",
    "all_device_kernels_per_call_us",
    "all_kernel_dur_sum_us",
]


def check_comparison(filepath: str, check_routes: bool = True):
    errors = []
    warnings = []

    if not os.path.exists(filepath):
        return {"status": "FAIL", "errors": [f"File not found: {filepath}"]}

    with open(filepath) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return {"status": "FAIL", "errors": [f"Invalid JSON: {e}"]}

    perf = data.get("performance", data.get("profiler_data", {}))
    if not perf:
        errors.append("No performance/profiler_data section found")
        return {"status": "FAIL", "errors": errors}

    checked_any_batch = False
    for batch_key, batch_data in perf.items():
        if not isinstance(batch_data, dict):
            continue
        if check_routes:
            for route in REQUIRED_ROUTES:
                if route not in batch_data:
                    warnings.append(f"Batch {batch_key}: missing route '{route}'")
                    continue
                route_data = batch_data[route]
                for field in REQUIRED_BATCH_FIELDS:
                    if field not in route_data:
                        warnings.append(
                            f"Batch {batch_key}/{route}: missing REQUIRED field '{field}'"
                        )
                found_any_all_dev = any(
                    f in route_data for f in OPTIONAL_BATCH_FIELDS
                )
                if not found_any_all_dev:
                    warnings.append(
                        f"Batch {batch_key}/{route}: no all_device_kernels field found "
                        f"(expected one of {OPTIONAL_BATCH_FIELDS})"
                    )
        checked_any_batch = True

    if not checked_any_batch:
        errors.append("No batch data found in performance/profiler_data")

    correctness = data.get("correctness")
    if correctness:
        if isinstance(correctness, dict):
            for route in REQUIRED_ROUTES:
                if route not in correctness:
                    warnings.append(f"Correctness section missing route '{route}'")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "file": filepath,
        "operator": data.get("operator", data.get("experiment", {}).get("operator", "unknown")),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate final_comparison.json")
    parser.add_argument("file", help="Path to final_comparison.json")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-route-check", action="store_true",
                        help="Skip per-route field checking")
    args = parser.parse_args()

    result = check_comparison(args.file, check_routes=not args.no_route_check)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[{result['status']}] {args.file}")
        if result.get("operator"):
            print(f"  Operator: {result['operator']}")
        for e in result.get("errors", []):
            print(f"  ERROR: {e}")
        for w in result.get("warnings", []):
            print(f"  WARN: {w}")

    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
