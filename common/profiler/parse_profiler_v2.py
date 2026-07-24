#!/usr/bin/env python3
"""Unified msprof trace parser v2 — corrects primary_compute_kernel_us as per-call mean."""
import os, sys, json, glob
from collections import defaultdict

COMPUTE_KERNEL_PREFIXES = ('KERNEL_AIVEC', 'KERNEL_AIC', 'KERNEL_MIX_AIC', 'KERNEL_CUBE')
RUNTIME_KERNEL_PREFIXES = ('KERNEL_AICPU',)

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

def classify_kernel(tt):
    for pfx in COMPUTE_KERNEL_PREFIXES:
        if tt.startswith(pfx):
            return 'compute'
    return 'runtime'

def main():
    logical_calls = None
    positional = []
    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1 + i]
        if arg == '--logical-calls' and i + 1 < len(sys.argv[1:]):
            logical_calls = int(sys.argv[1 + i + 1]); i += 2
        elif not arg.startswith('--'):
            positional.append(arg); i += 1
        else:
            i += 1
    if len(positional) < 1:
        print("Usage: parse_profiler_v2.py <raw_dir> [output_json] [--logical-calls N]", file=sys.stderr)
        sys.exit(1)
    raw_dir = positional[0]; output = positional[1] if len(positional) > 1 else None

    prof_dir = find_prof_dir(raw_dir)
    if not prof_dir:
        print(f"error: no PROF_* directory found in {raw_dir}"); sys.exit(1)

    trace_path = find_trace_file(prof_dir)
    if not trace_path:
        print(f"error: no msprof_*.json found under {prof_dir}"); sys.exit(1)

    with open(trace_path) as f:
        data = json.load(f)
    traces = data if isinstance(data, list) else data.get('traceEvents', [])

    kernel_events = []; host_events = []; aicpu_init_events = []
    for ev in traces:
        if not isinstance(ev, dict): continue
        if ev.get('ph') != 'X': continue
        name = ev.get('name', ''); dur = ev.get('dur', 0)
        tt = ev.get('args', {}).get('Task Type', 'N/A')
        if tt.startswith('KERNEL_'):
            if tt == 'KERNEL_AICPU' and name == 'KERNEL_AICPU':
                aicpu_init_events.append({'name': name, 'type': tt, 'dur_us': dur})
            else:
                kernel_events.append({'name': name, 'type': tt, 'dur_us': dur, 'class': classify_kernel(tt)})
        elif tt == 'N/A' and dur > 0:
            host_events.append({'name': name, 'dur_us': dur})

    total_dur = sum(e['dur_us'] for e in kernel_events)
    compute_events = [e for e in kernel_events if e['class'] == 'compute']
    runtime_events = [e for e in kernel_events if e['class'] == 'runtime']

    by_type = defaultdict(list)
    for e in kernel_events: by_type[e['type']].append(e['dur_us'])

    # Deduce logical call count
    if logical_calls:
        loop_count = logical_calls
    else:
        print("WARNING: --logical-calls not provided, defaulting to 100 loops", file=sys.stderr)
        loop_count = 100

    # Per-call total device kernel time
    all_device_kernels_us_per_call = round(total_dur / loop_count, 3) if loop_count else 0

    # Primary compute kernel time: mean per call (NOT max single event)
    if compute_events:
        compute_mean = round(sum(e['dur_us'] for e in compute_events) / len(compute_events), 3)
    elif kernel_events:
        compute_mean = round(sum(e['dur_us'] for e in kernel_events) / len(kernel_events), 3)
    else:
        compute_mean = 0.0
    primary_compute_kernel_us = round(compute_mean, 3)

    # Primary kernel name/type identification (max is fine for naming)
    primary = max(compute_events, key=lambda e: e['dur_us']) if compute_events else (max(kernel_events, key=lambda e: e['dur_us']) if kernel_events else {})

    kernels_per_call = round(len(kernel_events) / loop_count, 2) if loop_count else 0

    breakdown = {}
    for k, v in by_type.items():
        per_call_this = round(sum(v) / loop_count, 3) if loop_count else 0
        breakdown[k] = {'count': len(v), 'total_us': round(sum(v), 2), 'mean_us': round(sum(v) / len(v), 3), 'per_call_us': per_call_this}

    compute_kernel_types = sorted(set(e['type'] for e in compute_events))
    executor_kernel_types = sorted(set(e['type'] for e in kernel_events if e['class'] != 'compute'))

    result = {
        'kernel_count': len(kernel_events),
        'logical_calls': loop_count,
        'kernels_per_logical_call': kernels_per_call,
        'all_device_kernels_us_per_call': all_device_kernels_us_per_call,
        'primary_compute_kernel_us': primary_compute_kernel_us,
        'primary_kernel_name': primary.get('name', ''),
        'primary_kernel_type': primary.get('type', ''),
        'kernel_names': sorted(set(e['name'] for e in kernel_events)),
        'kernel_types': sorted(set(e['type'] for e in kernel_events)),
        'kernel_type_breakdown': breakdown,
        'compute_kernel_types': compute_kernel_types,
        'executor_kernel_types': executor_kernel_types,
        'total_kernel_duration_all_iters_us': round(total_dur, 2),
        'aicpu_init_us': round(sum(e['dur_us'] for e in aicpu_init_events), 2),
        'host_events_count': len(host_events),
        'parser_version': '2.0',
        'profiler_type': 'msprof',
    }

    # Validation
    validation = []
    def add_check(name, status, values=None):
        validation.append({'check': name, 'status': status, 'values': values})
    add_check('primary_compute_kernel_us > 0', 'PASS' if primary_compute_kernel_us > 0 or len(kernel_events) == 0 else 'FAIL', primary_compute_kernel_us)
    p_ok = primary_compute_kernel_us <= all_device_kernels_us_per_call * 1.01
    add_check('primary <= all_device_per_call', 'PASS' if p_ok else 'FAIL', {'primary': primary_compute_kernel_us, 'all_device': all_device_kernels_us_per_call})
    add_check('logical_calls > 0', 'PASS' if loop_count > 0 else 'FAIL', loop_count)
    add_check('kernels_per_call > 0', 'PASS' if kernels_per_call > 0 else 'FAIL', kernels_per_call)
    all_pass = all(v['status'] == 'PASS' for v in validation)
    result['_validation'] = {'all_pass': all_pass, 'checks': validation}

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w') as f: json.dump(result, f, indent=2)
        print(f"parsed profiler data -> {output}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
