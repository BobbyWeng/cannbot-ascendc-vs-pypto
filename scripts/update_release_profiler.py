#!/usr/bin/env python3
"""
Scan operators/*/reports/parsed/*.json and update current_release.json
with profiler routes structure for all operators.
"""
import json, os, hashlib, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RELEASE = BASE / "reports" / "release" / "current_release.json"
OPERATORS = BASE / "operators"
BATCHES = [1, 2, 4, 8, 16, 32, 64]
ROUTES = ["torch", "ascendc", "pypto"]

def load_json(p):
    try: return json.loads(p.read_text())
    except: return None

def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    release = json.loads(RELEASE.read_text())
    changed = False

    for op_name, op_entry in release.get("operators", {}).items():
        op_dir = OPERATORS / op_name
        if not op_dir.is_dir():
            continue

        routes = {}
        for rk in ROUTES:
            profiler = {}
            has_data = False

            for batch in BATCHES:
                parsed_file = op_dir / "reports" / "parsed" / f"{rk}_b{batch}.json"
                if parsed_file.exists():
                    data = load_json(parsed_file)
                    if data:
                        primary = data.get("primary_compute_kernel_us")
                        if primary and primary > 0:
                            profiler[f"b{batch}_us"] = round(primary, 3)
                            has_data = True
                        profiler["kernel_type"] = data.get("primary_compute_type", profiler.get("kernel_type", ""))
                        profiler["kernel_name"] = (data.get("kernel_names") or [""])[0] if data.get("kernel_names") else profiler.get("kernel_name", "")
                        kpc = data.get("kernels_per_call", 0)
                        if kpc:
                            profiler["kernels_per_call"] = kpc

            # Also try event-based data from experiment_config or TASK_STATE
            task_state = load_json(op_dir / "TASK_STATE.json")
            if task_state:
                ev = task_state.get("event_measured", {})
                ev_key = f"{rk}_b1"
                if ev and ev.get(ev_key):
                    val = ev[ev_key].replace(" us", "")
                    try:
                        if not profiler.get("b1_us"):
                            profiler["b1_us"] = round(float(val), 1)
                            profiler["method"] = "torch.npu.Event"
                            has_data = True
                    except: pass

            if has_data:
                profiler["method"] = "msprof"
                routes[rk] = {"profiler": profiler}

        if routes:
            # Only set if routes have more data than current
            existing = op_entry.get("routes", {})
            new_count = sum(1 for r in routes.values() if r.get("profiler", {}).get("b1_us"))
            old_count = sum(1 for r in existing.values() if r.get("profiler", {}).get("b1_us"))
            if new_count > old_count:
                op_entry["routes"] = routes
                changed = True
                routes_str = ", ".join([f"{k}(b1={v['profiler'].get('b1_us','?')}us)" for k,v in routes.items()])
                print(f"  {op_name}: {routes_str}")

    if changed:
        release["generated_at"] = "2026-07-22T15:00:00Z"
        RELEASE.write_text(json.dumps(release, indent=2, ensure_ascii=False) + "\n")
        print(f"\nUpdated {RELEASE}")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    main()
