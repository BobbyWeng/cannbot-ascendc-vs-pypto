#!/usr/bin/env python3
"""Validate a parsed profiler JSON file."""
import json
import sys
import os

REQUIRED_FIELDS = [
    "primary_compute_kernel_us",
    "all_device_kernels_us_per_call",
    "kernel_count",
    "kernels_per_logical_call",
    "kernel_names",
    "primary_kernel_name",
    "primary_kernel_type",
]

FIELD_CONSTRAINTS = {
    "primary_compute_kernel_us": {"min": 0},
    "kernel_count": {"min": 1},
    "kernels_per_logical_call": {"min": 0.5},
    "kernel_names": {"min_length": 1},
}


def _emit(is_new_format, errors, warnings, msg):
    if is_new_format:
        errors.append(msg)
    else:
        warnings.append(msg)


def check_parsed(filepath: str):
    errors = []
    warnings = []

    if not os.path.exists(filepath):
        return {"status": "FAIL", "errors": [f"File not found: {filepath}"]}

    with open(filepath) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return {"status": "FAIL", "errors": [f"Invalid JSON: {e}"]}

    is_new_format = "parser_version" in data
    is_error_state = data.get("error") is not None

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")
            continue
        val = data[field]
        constraint = FIELD_CONSTRAINTS.get(field, {})
        if "min" in constraint and isinstance(val, (int, float)):
            if val <= constraint["min"] and not is_error_state:
                errors.append(
                    f"Field {field} = {val} should be > {constraint['min']}"
                )
        if "min_length" in constraint and isinstance(val, (list, str)):
            if len(val) < constraint["min_length"] and not is_error_state:
                errors.append(
                    f"Field {field} has length {len(val)}, expected >= {constraint['min_length']}"
                )

    if "logical_calls" not in data:
        _emit(is_new_format, errors, warnings, "Missing field: logical_calls")

    logical_calls = data.get("logical_calls")
    kernels_per_call = data.get("kernels_per_logical_call")
    kernel_count = data.get("kernel_count")
    all_device_kernels_us = data.get("all_device_kernels_us")
    all_device_kernels_us_per_call = data.get("all_device_kernels_us_per_call")

    if logical_calls and logical_calls > 0 and kernel_count and kernels_per_call:
        expected_kpc = kernel_count / logical_calls
        if abs(expected_kpc - kernels_per_call) >= 0.01:
            _emit(is_new_format, errors, warnings,
                  f"kernel_count/logical_calls ({expected_kpc:.3f}) != kernels_per_logical_call ({kernels_per_call})")

    if (all_device_kernels_us is not None and all_device_kernels_us_per_call is not None
            and logical_calls and logical_calls > 0):
        expected_adk = all_device_kernels_us / logical_calls
        if abs(expected_adk - all_device_kernels_us_per_call) >= 0.01:
            _emit(is_new_format, errors, warnings,
                  f"all_device_kernels_us/logical_calls ({expected_adk:.3f}) != all_device_kernels_us_per_call ({all_device_kernels_us_per_call})")

    if "kernel_type_breakdown" in data:
        breakdown = data["kernel_type_breakdown"]
        total_count = sum(
            v.get("count", 0) for v in breakdown.values()
        )
        if total_count != data.get("kernel_count", 0):
            _emit(is_new_format, errors, warnings,
                  f"kernel_count={data.get('kernel_count')} != breakdown sum={total_count}")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "file": filepath,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate parsed profiler JSON")
    parser.add_argument("file", help="Path to parsed JSON file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = check_parsed(args.file)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[{result['status']}] {args.file}")
        for e in result.get("errors", []):
            print(f"  ERROR: {e}")
        for w in result.get("warnings", []):
            print(f"  WARN: {w}")

    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
