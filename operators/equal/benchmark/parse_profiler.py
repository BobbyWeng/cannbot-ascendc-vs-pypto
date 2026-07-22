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
    aicpu_init_events = []

    for ev in traces:
        if not isinstance(ev, dict):
            continue
        if ev.get('ph') != 'X':
            continue
        name = ev.get('name', '')
        dur = ev.get('dur', 0)
        tt = ev.get('args', {}).get('Task Type', 'N/A')

        if tt.startswith('KERNEL_'):
            # Identify first KERNEL_AICPU as init (it has name "KERNEL_AICPU")
            if tt == 'KERNEL_AICPU' and name == 'KERNEL_AICPU':
                aicpu_init_events.append({'name': name, 'type': tt, 'dur_us': dur})
            else:
                kernel_events.append({'name': name, 'type': tt, 'dur_us': dur})
        elif tt == 'N/A' and dur > 0:
            host_events.append({'name': name, 'dur_us': dur})

    total_dur = sum(e['dur_us'] for e in kernel_events)

    compute_types = ('KERNEL_AIVEC', 'KERNEL_AIC', 'KERNEL_MIX_AIC', 'KERNEL_CUBE')
    compute_events = [e for e in kernel_events if any(e['type'].startswith(t) for t in compute_types)]

    runtime_types = ('KERNEL_AICPU',)
    runtime_events = [e for e in kernel_events if any(e['type'].startswith(t) for t in runtime_types)]

    # Primary: pick the longest compute event. If no compute events, fall back to runtime.
    # Compute events (MIX_AIC, AIVEC, CUBE) represent actual NPU compute kernels.
    primary = max(compute_events, key=lambda e: e['dur_us']) if compute_events else (
        max(kernel_events, key=lambda e: e['dur_us']) if kernel_events else {}
    )

    # The AICPU events include JIT dispatch overhead. For bandwidth calculation,
    # use MIX_AIC events (the actual vector compute kernels) as the compute metric.
    mix_aic_events = [e for e in kernel_events if e['type'] == 'KERNEL_MIX_AIC']
    aivec_events = [e for e in kernel_events if e['type'] == 'KERNEL_AIVEC']
    cube_events = [e for e in kernel_events if e['type'] == 'KERNEL_CUBE']
    compute_kernel_list = mix_aic_events or aivec_events or cube_events

    by_type = defaultdict(list)
    for e in kernel_events:
        by_type[e['type']].append(e['dur_us'])

    # Deduce loop count: count iterations in the MIX_AIC or AIVEC pattern
    loop_count = 100
    for t in ('KERNEL_MIX_AIC', 'KERNEL_AIVEC', 'KERNEL_AIC', 'KERNEL_CUBE'):
        if t in by_type:
            loop_count = len(by_type[t])
            break

    # Average compute kernel duration (mean of MIX_AIC/AIVEC/CUBE events across all iterations)
    compute_list = mix_aic_events or aivec_events or cube_events or kernel_events
    compute_mean = round(sum(e['dur_us'] for e in compute_list) / max(len(compute_list), 1), 3) if compute_list else 0

    # Per-call stats
    per_call_us = round(total_dur / loop_count, 3) if loop_count else 0

    # by_type breakdown
    breakdown = {}
    for k, v in by_type.items():
        per_call_this = round(sum(v) / loop_count, 3) if loop_count else 0
        breakdown[k] = {
            'count': len(v),
            'total_us': round(sum(v), 2),
            'mean_us': round(sum(v) / len(v), 3),
            'per_call_us': per_call_this,
        }

    all_types = sorted(set(e['type'] for e in kernel_events))
    compute_kernel_types = sorted(set(
        e['type'] for e in kernel_events
        if any(e['type'].startswith(t) for t in compute_types)
    ))
    executor_kernel_types = sorted(set(
        e['type'] for e in kernel_events
        if not any(e['type'].startswith(t) for t in compute_types)
    ))

    result = {
        'kernel_count': len(kernel_events),
        'logical_calls': loop_count,
        'kernels_per_logical_call': round(len(kernel_events) / loop_count, 2) if loop_count else 0,
        'all_device_kernels_us_per_call': per_call_us,
        'primary_compute_kernel_us': round(primary.get('dur_us', 0) / 1, 3),
        'primary_kernel_name': primary.get('name', ''),
        'primary_kernel_type': primary.get('type', ''),
        'kernel_names': sorted(set(e['name'] for e in kernel_events)),
        'kernel_types': all_types,
        'kernel_type_breakdown': breakdown,
        'compute_kernel_types': compute_kernel_types,
        'executor_kernel_types': executor_kernel_types,
        'total_kernel_duration_all_iters_us': round(total_dur, 2),
        'aicpu_init_us': round(sum(e['dur_us'] for e in aicpu_init_events), 2),
        'host_events_count': len(host_events),
        'parser_version': '1.2-pypto',
        'profiler_type': 'msprof',
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
