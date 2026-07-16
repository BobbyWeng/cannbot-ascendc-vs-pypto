#!/usr/bin/env python3
"""Minimal profiler output parser for MatMul."""
import os, sys, json, re

def parse_msprof_output(raw_dir):
    """Extract kernel info from msprof output."""
    result = {"kernel_names": [], "kernel_types": [], "kernel_count": 0}
    timeline_file = os.path.join(raw_dir, "timeline.json")
    if os.path.exists(timeline_file):
        with open(timeline_file) as f:
            data = json.load(f)
        for event in data:
            if "kernel_name" in event:
                result["kernel_names"].append(event["kernel_name"])
                result["kernel_types"].append(event.get("kernel_type", "unknown"))
        result["kernel_count"] = len(result["kernel_names"])
    return result

def main():
    if len(sys.argv) < 3:
        print("Usage: parse_profiler.py <raw_dir> <output_path>")
        sys.exit(1)
    raw_dir = sys.argv[1]
    output_path = sys.argv[2]
    result = parse_msprof_output(raw_dir)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Parsed profiler data saved to {output_path}")

if __name__ == "__main__":
    main()
