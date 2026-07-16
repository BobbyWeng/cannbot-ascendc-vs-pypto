#!/usr/bin/env python3
"""Parse msprof raw profiler data for ReduceSum."""
import os
import sys
import json
import glob

OP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(OP_DIR, "reports", "raw")
PARSED_DIR = os.path.join(OP_DIR, "reports", "parsed")
os.makedirs(PARSED_DIR, exist_ok=True)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
VARIANTS = ["ascendc", "torch"]


def parse_variant(variant):
    results = {}
    for b in BATCHES:
        raw_path = os.path.join(RAW_DIR, f"{variant}_b{b}")
        if not os.path.exists(raw_path):
            results[b] = {"status": "NO_DATA", "raw_path": raw_path}
            continue

        timeline_files = glob.glob(os.path.join(raw_path, "**", "*.csv"), recursive=True)
        result = {
            "batch": b,
            "variant": variant,
            "timeline_files": timeline_files,
            "primary_compute_kernel_us": None,
            "all_device_kernels_us_per_call": None,
            "host_synchronized_operation_us": None,
            "kernel_names": [],
            "kernel_counts": {},
        }
        results[b] = result

    return results


def main():
    all_data = {}
    for variant in VARIANTS:
        all_data[variant] = parse_variant(variant)

    out_path = os.path.join(PARSED_DIR, "parsed_profiler_results.json")
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Parsed profiler results saved to {out_path}")


if __name__ == "__main__":
    main()
