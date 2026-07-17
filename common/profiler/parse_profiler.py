#!/usr/bin/env python3
"""Unified profiler parser for Cannbot."""
import os, sys, json, glob
from collections import defaultdict


def find_prof_dir(raw_dir):
    for d in sorted(os.listdir(raw_dir), reverse=True):
        if d.startswith('PROF_'):
            return os.path.join(raw_dir, d)
    return None


def find_task_time_csv(prof_dir):
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            if f.startswith('task_time') and f.endswith('.csv'):
                return os.path.join(root, f)
    return None


def find_trace_file(prof_dir):
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            if f.startswith('msprof_') and f.endswith('.json'):
                return os.path.join(root, f)
    return None


def parse_trace_json(trace_path, loops=100):
    with open(trace_path) as f:
        data = json.load(f)
    traces = data if isinstance(data, list) else data.get('traceEvents', [])
    kernel_events = []
    host_api_events = []
    for ev in traces:
        if not isinstance(ev, dict) or ev.get('ph') != 'X':
            continue
        name = ev.get('name', '')
        dur = ev.get('dur', 0)
        args = ev.get('args', {}) or {}
        tt = args.get('Task Type', 'N/A')
        if tt.startswith('KERNEL_'):
            kernel_events.append({'name': name, 'type': tt, 'dur_us': dur})
        elif tt == 'N/A' and dur > 0 and 'aclrt' in name:
            host_api_events.append({'name': name, 'dur_us': dur})

    if not kernel_events:
        return {"error": "no kernel events found", "kernel_count": 0}

    # Group by type
    by_type = defaultdict(list)
    for e in kernel_events:
        by_type[e['type']].append(e['dur_us'])

    total_dur = sum(e['dur_us'] for e in kernel_events)

    # Find primary compute kernel: prefer non-AICPU compute kernels over AICPU executors
    compute_kernels = [e for e in kernel_events if 'AICPU' not in e['type']]
    if compute_kernels:
        primary = max(compute_kernels, key=lambda e: e['dur_us'])
    else:
        primary = max(kernel_events, key=lambda e: e['dur_us'])

    # Count unique compute kernel types (not AICPU)
    compute_types = [t for t in by_type if 'AICPU' not in t]
    executor_types = [t for t in by_type if 'AICPU' in t]

    kernels_per_call = round(len(kernel_events) / loops, 2) if loops > 0 else 0
    all_required_device_kernels_us = round(total_dur / loops, 3) if loops > 0 else 0

    result = {
        'kernel_count': len(kernel_events),
        'logical_calls': loops,
        'kernels_per_logical_call': kernels_per_call,
        'all_device_kernels_us_per_call': all_required_device_kernels_us,
        'primary_compute_kernel_us': round(primary['dur_us'], 3),
        'primary_kernel_name': primary['name'],
        'primary_kernel_type': primary['type'],
        'kernel_names': sorted(set(e['name'] for e in kernel_events)),
        'kernel_types': sorted(set(e['type'] for e in kernel_events)),
        'kernel_type_breakdown': {
            k: {
                'count': len(v),
                'total_us': round(sum(v), 2),
                'mean_us': round(sum(v) / len(v), 3),
                'per_call_us': round(sum(v) / loops, 3) if loops > 0 else 0
            }
            for k, v in by_type.items()
        },
        'compute_kernel_types': sorted(compute_types),
        'executor_kernel_types': sorted(executor_types) if executor_types else [],
        'total_kernel_duration_all_iters_us': round(total_dur, 2),
    }

    # AICPU runtime tracking
    if executor_types:
        aicpu_total = sum(sum(v) for k, v in by_type.items() if k in executor_types)
        result['aicpu_executor_us_per_call'] = round(aicpu_total / loops, 3) if loops > 0 else 0

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_profiler.py <raw_dir> [output_json] [--loops N]")
        sys.exit(1)

    raw_dir = sys.argv[1]
    output = None
    loops = 100
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--loops' and i + 1 < len(args):
            loops = int(args[i + 1])
            i += 2
        else:
            output = args[i]
            i += 1

    prof_dir = find_prof_dir(raw_dir)
    if not prof_dir:
        print(f"error: no PROF_* directory found in {raw_dir}")
        sys.exit(1)

    trace_path = find_trace_file(prof_dir)
    if not trace_path:
        print(f"error: no msprof_*.json found under {prof_dir}")
        sys.exit(1)

    result = parse_trace_json(trace_path, loops=loops)

    if output:
        out_dir = os.path.dirname(output) if os.path.dirname(output) else '.'
        os.makedirs(out_dir, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"parsed -> {output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
