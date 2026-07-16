#!/usr/bin/env python3
import os, sys, json, glob
from collections import defaultdict


def find_prof_dir(raw_dir):
    for d in sorted(os.listdir(raw_dir), reverse=True):
        if d.startswith('PROF_'):
            return os.path.join(raw_dir, d)
    return None


def find_trace_file(prof_dir):
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            if f.startswith('msprof_') and f.endswith('.json'):
                return os.path.join(root, f)
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_profiler.py <raw_dir> [output_json]")
        sys.exit(1)

    raw_dir = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    prof_dir = find_prof_dir(raw_dir)
    if not prof_dir:
        print(f"error: no PROF_* directory found in {raw_dir}")
        sys.exit(1)

    trace_path = find_trace_file(prof_dir)
    if not trace_path:
        print(f"error: no msprof_*.json found under {prof_dir}")
        sys.exit(1)

    with open(trace_path) as f:
        data = json.load(f)

    traces = data if isinstance(data, list) else data.get('traceEvents', [])

    kernel_events = []
    host_events = []
    for ev in traces:
        if not isinstance(ev, dict):
            continue
        if ev.get('ph') != 'X':
            continue
        name = ev.get('name', '')
        dur = ev.get('dur', 0)
        tt = ev.get('args', {}).get('Task Type', 'N/A')
        if tt.startswith('KERNEL_'):
            kernel_events.append({'name': name, 'type': tt, 'dur_us': dur})
        elif tt == 'N/A' and dur > 0:
            host_events.append({'name': name, 'dur_us': dur})

    total_dur = sum(e['dur_us'] for e in kernel_events)
    primary = max(kernel_events, key=lambda e: e['dur_us']) if kernel_events else {}

    by_type = defaultdict(list)
    for e in kernel_events:
        by_type[e['type']].append(e['dur_us'])

    loops = 100
    kernels_per_call = round(len(kernel_events) / loops, 2) if loops > 0 else 0
    all_device_kernels_us = round(total_dur / loops, 3) if loops > 0 else 0

    result = {
        'kernel_count': len(kernel_events),
        'kernels_per_call': kernels_per_call,
        'all_device_kernels_us': all_device_kernels_us,
        'primary_compute_kernel_us': round(primary.get('dur_us', 0) / 1, 3),
        'primary_compute_type': primary.get('type', ''),
        'kernel_names': sorted(set(e['name'] for e in kernel_events)),
        'by_type': {
            k: {
                'count': len(v),
                'total_us': round(sum(v), 2),
                'mean_us': round(sum(v) / len(v), 3)
            }
            for k, v in by_type.items()
        },
        'host_events_count': len(host_events),
    }

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"parsed profiler data -> {output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
