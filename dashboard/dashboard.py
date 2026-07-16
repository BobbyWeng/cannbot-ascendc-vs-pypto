#!/usr/bin/env python3
"""PyPTO Operator Dashboard — Auto-scan & Static HTML Generator."""

import json
import os
import re
import glob as glob_mod
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
OPERATORS = BASE / "operators"
OUT = BASE / "dashboard"


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_yaml(path):
    try:
        import yaml
        return yaml.safe_load(path.read_text())
    except (ImportError, FileNotFoundError, Exception):
        return None


def safe_float(v, default=None):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def get(op_name):
    d = OPERATORS / op_name
    data = {"name": op_name, "status": "unknown", "formula": "N/A",
            "shape": "N/A", "dtype": "N/A", "batches": [],
            "precision": "N/A", "broadcast": False,
            "kernel_shape": "N/A", "logical_shape": "N/A",
            "torch": {}, "ascendc": {}, "pypto": {},
            "correctness": {}, "correctness_all_pass": None,
            "orchestrator": {}, "dev_status": {},
            "profiler": {}, "history": [],
            "last_update": "N/A", "fastest": "N/A",
            "kernel_types": {}, "kernel_count": {}}

    # SPEC
    for fname in ("SPEC.yaml", "SPEC.yml", "spec.yaml", "spec.yml"):
        spec = load_yaml(d / fname)
        if spec:
            break
    if not spec:
        spec = load_json(d / "SPEC.json") or load_json(d / "spec.json")
    if spec:
        data["formula"] = spec.get("formula", data["formula"])
        data["dtype"] = spec.get("dtype", "N/A")
        data["batches"] = spec.get("batches", [])
        if "precision" in spec and isinstance(spec["precision"], dict):
            p = spec["precision"]
            parts = []
            if p.get("rtol") is not None:
                parts.append(f"rtol={p['rtol']}")
            if p.get("atol") is not None:
                parts.append(f"atol={p['atol']}")
            if p.get("require_bitwise"):
                parts.append("bitwise")
            if p.get("note"):
                parts.append(f"({p['note']})")
            data["precision"] = ", ".join(parts) if parts else data["precision"]
        inputs = spec.get("inputs", [])
        outputs = spec.get("outputs", [])
        if inputs:
            inp = inputs[0]
            data["shape"] = inp.get("shape", inp.get("logical_shape", data["shape"]))
            data["kernel_shape"] = inp.get("kernel_shape", data["shape"])
            data["logical_shape"] = inp.get("logical_shape", data["shape"])
        if outputs:
            out = outputs[0]
            if out.get("shape") and data["shape"] == "N/A":
                data["shape"] = out["shape"]
        if spec.get("broadcast_axis") is not None or "broadcast" in spec.get("category", ""):
            data["broadcast"] = True

    # experiment_config
    exp = load_yaml(d / "experiment_config.yaml")
    if not exp:
        exp = load_yaml(d / "experiment_config.yml")
    if exp:
        data["torch"]["type"] = exp.get("implementations", {}).get("torch", {}).get("type", "N/A")
        data["ascendc"]["type"] = exp.get("implementations", {}).get("ascendc", {}).get("type", "N/A")
        data["pypto"]["type"] = exp.get("implementations", {}).get("pypto", {}).get("type", "N/A")

    # orchestrator state
    for sub in ("", "pypto/"):
        orch = load_json(d / f"{sub}.orchestrator_state.json")
        if orch:
            data["orchestrator"] = orch
            data["dev_status"] = orch.get("stage_status", {})
            break

    # final comparison
    final = load_json(d / "reports" / "final" / "final_comparison.json")
    if not final:
        final = load_json(d / "reports" / "final" / "comparison_results.json")

    if final:
        data["formula"] = final.get("formula", final.get("description", data["formula"]))
        data["shape"] = final.get("shape", data["shape"])
        data["dtype"] = final.get("dtype", data["dtype"])
        data["batches"] = final.get("experiment", final).get("batches", data["batches"])

        # correctness
        corr = final.get("correctness", {})
        if isinstance(corr, dict):
            all_pass = corr.get("all_batches_pass")
            if all_pass is not None:
                data["correctness_all_pass"] = all_pass
            else:
                status_val = corr.get("status")
                if status_val:
                    data["correctness_all_pass"] = status_val == "PASS"
            data["correctness"]["note"] = corr.get("note", "")
        elif isinstance(corr, str):
            data["correctness_all_pass"] = "PASS" in corr.upper()
            data["correctness"]["note"] = corr
        # Check per-implementation correctness (add comparison_results.json style)
        if data["correctness_all_pass"] is None and "results" in final:
            results = final["results"]
            impl_statuses = []
            for impl_name, impl_data in results.items():
                c = impl_data.get("correctness", "")
                if isinstance(c, str) and "PASS" in c.upper():
                    impl_statuses.append(True)
                elif isinstance(c, str) and "FAIL" in c.upper():
                    impl_statuses.append(False)
            if impl_statuses:
                data["correctness_all_pass"] = all(impl_statuses)
        # Check div style: per-implementation dict
        if data["correctness_all_pass"] is None:
            for impl_name in ("torch", "ascendc", "pypto"):
                c = final.get("correctness", {}).get(impl_name, {})
                if isinstance(c, dict):
                    st = c.get("status", "")
                    if st == "PASS":
                        data["correctness_all_pass"] = True
                    elif st == "FAIL":
                        data["correctness_all_pass"] = False

        # perf data
        pdata = final.get("profiler_data", {})
        summary = final.get("comparison_summary", [])
        if not pdata and "results" in final:
            results = final["results"]
            for impl in ("torch", "ascendc", "pypto"):
                if impl in results:
                    batches_data = results[impl].get("batches", {})
                    for bk, bv in batches_data.items():
                        if bk not in pdata:
                            pdata[bk] = {}
                        if impl not in pdata[bk]:
                            pdata[bk][impl] = {}
                        pdata[bk][impl]["primary_compute_kernel_us"] = bv.get("median_us")
                        pdata[bk][impl]["all_device_kernels_us"] = bv.get("median_us")

        data["profiler_data"] = pdata
        data["comparison_summary"] = summary

        for bk, bv in pdata.items():
            for impl in ("torch", "ascendc"):
                if impl in bv:
                    d_impl = bv[impl]
                    if impl not in data["kernel_types"]:
                        data["kernel_types"][impl] = set()
                    kt = d_impl.get("kernel_type", "N/A")
                    data["kernel_types"][impl].add(kt)
                    kc = d_impl.get("kernels_per_call", 0)
                    if isinstance(kc, dict):
                        kc = kc.get("total", 0)
                    data["kernel_count"][impl] = kc
            if "pypto" in bv:
                dp = bv["pypto"]
                if "pypto" not in data["kernel_types"]:
                    data["kernel_types"]["pypto"] = set()
                pk = dp.get("primary_kernel_type", dp.get("kernel_type", "N/A"))
                ek = dp.get("executor_kernel_type", "")
                data["kernel_types"]["pypto"].add(pk)
                if ek:
                    data["kernel_types"]["pypto"].add(ek)
                kc = dp.get("kernels_per_call", 0)
                if isinstance(kc, dict):
                    kc = kc.get("total", 0)
                data["kernel_count"]["pypto"] = kc

        if summary and isinstance(summary, list):
            fastest_counts = {}
            for s in summary:
                f = s.get("fastest", "")
                if f:
                    fastest_counts[f] = fastest_counts.get(f, 0) + 1
            if fastest_counts:
                data["fastest"] = max(fastest_counts, key=fastest_counts.get)

    # raw unified summary
    u = load_json(d / "reports" / "raw" / "unified_summary.json")
    if u:
        results = u.get("results", {})
        for bk, bv in results.items():
            impls = bv.get("implementations", {})
            for impl_key, iv in impls.items():
                if "kernel" in iv:
                    k = iv["kernel"]
                    label = "ascendc" if "ascendc" in impl_key else ("pypto" if "pypto" in impl_key else "torch")
                    if label not in data:
                        data[label] = {}
                    if "benchmark" not in data[label]:
                        data[label]["benchmark"] = {}
                    data[label]["benchmark"][bk] = {
                        "median_us": k.get("median_us"),
                        "mean_us": k.get("mean_us"),
                        "min_us": k.get("min_us"),
                        "std_us": k.get("std_us"),
                        "cv_percent": k.get("cv_percent"),
                        "bw_GBps": k.get("bw_GBps"),
                    }

    # correctness from unified summary
    if u and "correctness" in u:
        for c in u["correctness"]:
            label = "ascendc" if "ascendc" in c.get("implementation", "") else ("pypto" if "pypto" in c.get("implementation", "") else "torch")
            bk = str(c.get("batch", ""))
            if label not in data["correctness"]:
                data["correctness"][label] = {}
            data["correctness"][label][bk] = {
                "status": c.get("status", "N/A"),
                "bitwise_equal": c.get("bitwise_equal"),
                "max_abs_diff": c.get("max_abs_diff"),
                "bitwise_mismatch_count": c.get("bitwise_mismatch_count"),
                "signed_zero_mismatch_count": c.get("signed_zero_mismatch_count"),
                "numeric_mismatch_count": c.get("numeric_mismatch_count"),
                "nan_count": c.get("nan_count"),
                "inf_count": c.get("inf_count"),
            }

    # parsed reports (profiler detail — summary only, skip raw events to keep JSON small)
    parsed_dir = d / "reports" / "parsed"
    if parsed_dir.exists():
        parsed_files = sorted(parsed_dir.glob("*.json"))
        data["parsed"] = {}
        for pf in parsed_files:
            pdata_parsed = load_json(pf)
            if pdata_parsed:
                stem = pf.stem
                # Only keep summary stats, drop raw kernel_events
                summary = {
                    "kernel_count": pdata_parsed.get("kernel", {}).get("kernel_count", 0),
                    "total_kernel_dur_us": pdata_parsed.get("kernel", {}).get("total_kernel_dur_us", 0),
                    "by_type": pdata_parsed.get("kernel", {}).get("by_type", {}),
                    "by_name": pdata_parsed.get("kernel", {}).get("by_name", {}),
                    "kernels_per_call": pdata_parsed.get("kernel", {}).get("kernels_per_call", 0),
                    "all_kernel_dur_per_call_us": pdata_parsed.get("kernel", {}).get("all_kernel_dur_per_call_us", 0),
                }
                data["parsed"][stem] = summary

    # history
    history_dir = d / "reports" / "history"
    if history_dir.exists():
        hfiles = sorted(history_dir.glob("v*/final_comparison.json")) + sorted(history_dir.glob("v*/comparison_results.json"))
        for hf in hfiles:
            hdata = load_json(hf)
            if hdata:
                version = hf.parent.name
                data["history"].append({"version": version, "data": hdata})

    # last update
    latest_mtime = 0
    for f in d.rglob("*"):
        if f.is_file() and not f.name.startswith("."):
            try:
                mt = f.stat().st_mtime
                if mt > latest_mtime:
                    latest_mtime = mt
            except OSError:
                pass
    if latest_mtime:
        data["last_update"] = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M")

    # status
    st = data["orchestrator"].get("current_stage", 0)
    if st >= 7:
        data["status"] = "completed"
    elif st >= 1:
        data["status"] = "in_progress"
    else:
        # No orchestrator: infer from artifact existence
        has_torch = bool(load_json(d / "torch" / "benchmark_results.json"))
        has_ascendc = (d / "ascendc" / "build").exists() and any((d / "ascendc" / "build").iterdir())
        has_pypto_golden = (d / "pypto" / "golden").exists()
        has_final_report = bool(final)
        if has_final_report:
            data["status"] = "completed" if data.get("correctness_all_pass") is True else "in_progress"
        elif has_torch and has_ascendc:
            data["status"] = "in_progress"
        else:
            data["status"] = "planned"

    locked = data["orchestrator"].get("lock_status", "")
    if "BLOCKED" in locked:
        data["status"] = "blocked"

    # dev status detail
    dev = data["dev_status"]
    sd = {"Intent": "completed" if dev.get("1") == "completed" else "pending",
          "API": "completed" if dev.get("2") == "completed" else "pending",
          "Golden": "completed" if dev.get("3") == "completed" else "pending",
          "Design": "completed" if dev.get("4") == "completed" else "pending",
          "Implementation": "completed" if dev.get("5") == "completed" else "pending",
          "Correctness": "completed" if dev.get("6") == "completed" else "pending",
          "Benchmark": "completed" if dev.get("7") == "completed" else "pending",
          "Archive": "completed"}
    for k in sd:
        if dev.get(k.lower()):
            sd[k] = dev[k.lower()]
    data["dev_status_detail"] = sd

    # Convert sets to lists for JSON
    for impl in ("torch", "ascendc", "pypto"):
        if impl in data["kernel_types"]:
            data["kernel_types"][impl] = sorted(data["kernel_types"][impl])

    return data


def build():
    ops_dirs = sorted([
        d.name for d in OPERATORS.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
    ])

    operators = []
    for name in ops_dirs:
        operators.append(get(name))

    # Project summary
    total = len(operators)
    completed = sum(1 for o in operators if o["status"] == "completed")
    torch_done = sum(1 for o in operators if "KERNEL_AIVEC" in str(o["kernel_types"].get("torch", [])))
    ascendc_done = sum(1 for o in operators if "KERNEL_AIVEC" in str(o["kernel_types"].get("ascendc", [])))
    pypto_done = sum(1 for o in operators if o["pypto"])
    pass_count = sum(1 for o in operators if o["correctness_all_pass"] is True)
    fail_count = sum(1 for o in operators if o["correctness_all_pass"] is False)
    blocked = sum(1 for o in operators if o["status"] == "blocked")

    dashboard = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": total,
            "completed": completed,
            "torch_done": torch_done,
            "ascendc_done": ascendc_done,
            "pypto_done": pypto_done,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "blocked": blocked,
        },
        "operators": operators,
    }

    # Write dashboard.json
    out_json = OUT / "dashboard.json"
    OUT.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False, default=str))
    print(f"[OK] Written {out_json}")

    # Generate index.html
    html = generate_html(dashboard)
    out_html = OUT / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] Written {out_html}")

    # Write CSS/JS inline in HTML, but also keep standalone files
    css = generate_css()
    (OUT / "dashboard.css").write_text(css, encoding="utf-8")
    js = generate_js()
    (OUT / "dashboard.js").write_text(js, encoding="utf-8")

    print(f"\nDashboard ready: {OUT / 'index.html'}")
    print(f"  Total operators: {total}")
    print(f"  Completed: {completed}")
    print(f"  In progress: {total - completed - blocked}")
    print(f"  Blocked: {blocked}")
    print(f"  Correctness PASS: {pass_count}")
    print(f"  Correctness FAIL: {fail_count}")


def generate_css():
    return """/* PyPTO Dashboard — Dark Theme */
:root {
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-tertiary: #21262d;
  --bg-hover: #30363d;
  --border: #30363d;
  --text-primary: #e6edf3;
  --text-secondary: #8b949e;
  --text-muted: #6e7681;
  --accent-blue: #58a6ff;
  --accent-green: #3fb950;
  --accent-yellow: #d29922;
  --accent-red: #f85149;
  --accent-purple: #bc8cff;
  --accent-orange: #f0883e;
  --accent-cyan: #39d2c0;
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  min-height: 100vh;
}

a { color: var(--accent-blue); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Header */
.header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}
.header h1 {
  font-size: 20px;
  font-weight: 600;
}
.header .subtitle {
  color: var(--text-secondary);
  font-size: 13px;
  margin-top: 2px;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}
.header-right .update-time {
  color: var(--text-muted);
  font-size: 12px;
}
.header-right .badge {
  background: var(--accent-blue);
  color: #fff;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

/* Search & Sort bar */
.toolbar {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.toolbar input {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 12px;
  color: var(--text-primary);
  font-size: 14px;
  width: 240px;
  outline: none;
}
.toolbar input:focus {
  border-color: var(--accent-blue);
}
.toolbar select {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 12px;
  color: var(--text-primary);
  font-size: 14px;
  outline: none;
  cursor: pointer;
}

/* Container */
.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 24px;
}

/* Summary Cards */
.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.summary-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}
.summary-card .value {
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 4px;
}
.summary-card .label {
  font-size: 12px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.summary-card.color-blue .value { color: var(--accent-blue); }
.summary-card.color-green .value { color: var(--accent-green); }
.summary-card.color-red .value { color: var(--accent-red); }
.summary-card.color-yellow .value { color: var(--accent-yellow); }
.summary-card.color-purple .value { color: var(--accent-purple); }
.summary-card.color-orange .value { color: var(--accent-orange); }
.summary-card.color-cyan .value { color: var(--accent-cyan); }

/* Progress Bar */
.progress-bar {
  background: var(--bg-tertiary);
  border-radius: 6px;
  height: 8px;
  overflow: hidden;
  margin-top: 8px;
}
.progress-bar .fill {
  height: 100%;
  border-radius: 6px;
  transition: width 0.3s;
}

/* Table */
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--bg-secondary);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 24px;
}
th {
  background: var(--bg-tertiary);
  padding: 10px 14px;
  text-align: left;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
th:hover { color: var(--text-primary); }
th .sort-arrow { margin-left: 4px; opacity: 0.5; }
th.sorted .sort-arrow { opacity: 1; }
td {
  padding: 10px 14px;
  border-top: 1px solid var(--border);
  font-size: 14px;
}
tr { cursor: pointer; }
tr:hover { background: var(--bg-hover); }

/* Status badges */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}
.status-badge.completed { background: rgba(63,185,80,0.15); color: var(--accent-green); }
.status-badge.in_progress { background: rgba(210,153,34,0.15); color: var(--accent-yellow); }
.status-badge.planned { background: rgba(110,118,129,0.15); color: var(--text-muted); }
.status-badge.blocked { background: rgba(248,81,73,0.15); color: var(--accent-red); }
.status-badge.pass { background: rgba(63,185,80,0.15); color: var(--accent-green); }
.status-badge.fail { background: rgba(248,81,73,0.15); color: var(--accent-red); }
.status-badge.unknown { background: rgba(110,118,129,0.15); color: var(--text-muted); }

/* Detail view */
.detail-view {
  display: none;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 24px;
  margin-bottom: 24px;
}
.detail-view.visible { display: block; }
.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.detail-header h2 { font-size: 24px; }
.detail-close {
  background: none;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 4px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}
.detail-close:hover { background: var(--bg-hover); color: var(--text-primary); }

/* Info grid */
.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.info-item {
  background: var(--bg-tertiary);
  border-radius: 6px;
  padding: 12px 16px;
}
.info-item .label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}
.info-item .value {
  font-size: 15px;
  font-weight: 500;
}

/* Section */
.section {
  margin-bottom: 24px;
}
.section h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

/* Comparison table within detail */
.comparison-table th, .comparison-table td {
  font-size: 13px;
  text-align: center;
}
.comparison-table td:first-child {
  text-align: left;
  font-weight: 500;
}
.comparison-table .fastest {
  color: var(--accent-green);
  font-weight: 600;
}

/* Kernel timeline */
.kernel-timeline { margin-bottom: 24px; }
.kernel-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}
.kernel-label {
  width: 100px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  text-align: right;
}
.kernel-bar-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
}
.kernel-bar {
  height: 24px;
  border-radius: 4px;
  min-width: 4px;
  transition: width 0.3s;
  position: relative;
}
.kernel-bar .bar-label {
  position: absolute;
  left: 8px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 11px;
  white-space: nowrap;
  color: #fff;
  text-shadow: 0 1px 2px rgba(0,0,0,0.5);
}
.kernel-bar.torch { background: var(--accent-blue); }
.kernel-bar.ascendc { background: var(--accent-green); }
.kernel-bar.pypto { background: var(--accent-orange); }
.kernel-dur {
  width: 80px;
  font-size: 12px;
  color: var(--text-secondary);
  text-align: right;
}

/* Charts */
.chart-container {
  background: var(--bg-tertiary);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}
.chart-container canvas {
  max-height: 300px;
}

/* Correctness heatmap */
.heatmap {
  display: inline-grid;
  gap: 2px;
}
.heatmap-cell {
  width: 28px;
  height: 28px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 600;
}
.heatmap-cell.pass { background: rgba(63,185,80,0.3); color: var(--accent-green); }
.heatmap-cell.fail { background: rgba(248,81,73,0.3); color: var(--accent-red); }
.heatmap-cell.na { background: rgba(110,118,129,0.15); color: var(--text-muted); }

/* Dev status pipeline */
.pipeline {
  display: flex;
  gap: 4px;
  align-items: center;
  flex-wrap: wrap;
}
.pipeline .step {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
}
.pipeline .step.completed { background: rgba(63,185,80,0.15); color: var(--accent-green); }
.pipeline .step.in_progress { background: rgba(210,153,34,0.15); color: var(--accent-yellow); }
.pipeline .step.pending { background: rgba(110,118,129,0.1); color: var(--text-muted); }
.pipeline .step.failed { background: rgba(248,81,73,0.15); color: var(--accent-red); }
.pipeline .arrow { color: var(--text-muted); font-size: 11px; }

/* Kernel type tag */
.kernel-tag {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 500;
  margin: 1px;
}
.kernel-tag.aivec { background: rgba(88,166,255,0.2); color: var(--accent-blue); }
.kernel-tag.aicpu { background: rgba(240,136,62,0.2); color: var(--accent-orange); }
.kernel-tag.mix { background: rgba(188,140,255,0.2); color: var(--accent-purple); }
.kernel-tag.aic { background: rgba(57,210,192,0.2); color: var(--accent-cyan); }
.kernel-tag.default { background: rgba(110,118,129,0.15); color: var(--text-muted); }

/* Tab system */
.tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}
.tab {
  padding: 8px 16px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
}
.tab:hover { color: var(--text-primary); }
.tab.active { color: var(--accent-blue); border-bottom-color: var(--accent-blue); }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* History timeline */
.history-item {
  background: var(--bg-tertiary);
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 8px;
}
.history-item .version {
  font-weight: 600;
  color: var(--accent-blue);
}

/* Responsive */
@media (max-width: 768px) {
  .summary-cards { grid-template-columns: repeat(2, 1fr); }
  .info-grid { grid-template-columns: repeat(2, 1fr); }
  .toolbar input { width: 160px; }
}

/* Animations */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
.summary-card, .detail-view.visible { animation: fadeIn 0.3s ease-out; }

/* Pie chart legend */
.legend {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 8px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-secondary);
}
.legend-item .dot {
  width: 10px;
  height: 10px;
  border-radius: 2px;
}
"""


def generate_js():
    return """// PyPTO Dashboard — Interactive
let dashboardData = null;
let currentOp = null;
let currentTab = 'correctness';
let sortField = 'name';
let sortAsc = true;

function init() {
  fetch('./dashboard.json')
    .then(r => r.json())
    .then(data => {
      dashboardData = data;
      renderSummary(data);
      renderTable(data);
      setupSearch();
      setupSort();
      document.getElementById('loading').style.display = 'none';
      document.getElementById('app').style.display = 'block';
    })
    .catch(err => {
      document.getElementById('loading').textContent = 'Failed to load dashboard.json: ' + err.message;
    });
}

function renderSummary(data) {
  const s = data.summary;
  const cards = [
    { label: 'Total Operators', value: s.total, cls: 'color-blue' },
    { label: 'Completed', value: s.completed, cls: 'color-green' },
    { label: 'Torch Ready', value: s.torch_done, cls: 'color-cyan' },
    { label: 'Ascend C Ready', value: s.ascendc_done, cls: 'color-purple' },
    { label: 'PyPTO Ready', value: s.pypto_done, cls: 'color-orange' },
    { label: 'Correctness PASS', value: s.pass_count, cls: 'color-green' },
    { label: 'Correctness FAIL', value: s.fail_count, cls: 'color-red' },
    { label: 'Blocked', value: s.blocked, cls: 'color-yellow' },
  ];
  const container = document.getElementById('summary-cards');
  container.innerHTML = cards.map(c => `
    <div class="summary-card ${c.cls}">
      <div class="value">${c.value}</div>
      <div class="label">${c.label}</div>
    </div>
  `).join('');

  // Progress bar
  const pct = s.total > 0 ? Math.round(s.completed / s.total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent = pct + '% (' + s.completed + '/' + s.total + ')';
  document.getElementById('update-time').textContent = 'Generated: ' + data.generated_at;
}

function renderTable(data) {
  const ops = [...data.operators];
  ops.sort((a, b) => {
    let va = String(a[sortField] || '').toLowerCase();
    let vb = String(b[sortField] || '').toLowerCase();
    if (sortField === 'name') {
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    // For status
    if (sortField === 'status') {
      const order = { completed: 0, in_progress: 1, planned: 2, blocked: 3 };
      const da = order[a.status] || 99;
      const db = order[b.status] || 99;
      return sortAsc ? da - db : db - da;
    }
    return 0;
  });

  const tbody = document.getElementById('op-table-body');
  tbody.innerHTML = ops.map(op => {
    const corr = op.correctness_all_pass === true ? '<span class="status-badge pass">PASS</span>' :
                 op.correctness_all_pass === false ? '<span class="status-badge fail">FAIL</span>' :
                 '<span class="status-badge unknown">N/A</span>';

    const torchKt = op.kernel_types?.torch?.join(', ') || 'N/A';
    const ascendcKt = op.kernel_types?.ascendc?.join(', ') || 'N/A';
    const pyptoKt = op.kernel_types?.pypto?.join(', ') || 'N/A';

    const ktc = op.kernel_count || {};
    const kcTorch = ktc.torch ?? 'N/A';
    const kcAscendc = ktc.ascendc ?? 'N/A';
    const kcPypto = ktc.pypto ?? 'N/A';

    // Benchmark table data
    let tb = op.torch?.benchmark;
    let ab = op.ascendc?.benchmark;
    let pb = op.pypto?.benchmark;
    let perf1 = '';
    if (op.batches && op.batches.length > 0) {
      const b1 = String(op.batches[0]);
      const tLat = tb?.[b1]?.median_us != null ? tb[b1].median_us.toFixed(1) + 'μs' : '';
      const aLat = ab?.[b1]?.median_us != null ? ab[b1].median_us.toFixed(1) + 'μs' : '';
      const pLat = pb?.[b1]?.median_us != null ? pb[b1].median_us.toFixed(1) + 'μs' : '';
      // Try profiler data
      const pd = op.profiler_data?.[b1];
      if (pd) {
        const tf = pd.torch?.primary_compute_kernel_us;
        const af = pd.ascendc?.primary_compute_kernel_us;
        const pf = pd.pypto?.primary_compute_kernel_us;
        if (tf != null) perf1 += 'T:' + tf.toFixed(1) + 'μs ';
        if (af != null) perf1 += 'A:' + af.toFixed(1) + 'μs ';
        if (pf != null) perf1 += 'P:' + pf.toFixed(1) + 'μs';
      } else {
        if (tLat) perf1 += 'T:' + tLat + ' ';
        if (aLat) perf1 += 'A:' + aLat + ' ';
        if (pLat) perf1 += 'P:' + pLat;
      }
    } else if (op.comparison_summary && op.comparison_summary[0]) {
      const s0 = op.comparison_summary[0];
      const tf = s0.torch_primary_us || s0.torch_us;
      const af = s0.ascendc_primary_us || s0.ascendc_us;
      const pf = s0.pypto_primary_us || s0.pypto_compute_us;
      if (tf != null) perf1 += 'T:' + Number(tf).toFixed(1) + 'μs ';
      if (af != null) perf1 += 'A:' + Number(af).toFixed(1) + 'μs ';
      if (pf != null) perf1 += 'P:' + Number(pf).toFixed(1) + 'μs';
    }
    if (!perf1) perf1 = 'N/A';

    return `<tr onclick="showDetail('${op.name}')">
      <td><strong>${op.name}</strong></td>
      <td><span class="status-badge ${op.status}">${op.status}</span></td>
      <td>${torchKt}</td>
      <td>${ascendcKt}</td>
      <td>${pyptoKt}</td>
      <td>${corr}</td>
      <td style="font-size:12px">${perf1}</td>
      <td style="font-size:12px">${op.last_update}</td>
    </tr>`;
  }).join('');

  // Update sort arrows
  document.querySelectorAll('#op-table th').forEach(th => {
    const field = th.dataset.sort;
    th.classList.toggle('sorted', field === sortField);
    const arrow = th.querySelector('.sort-arrow');
    if (arrow && field === sortField) {
      arrow.textContent = sortAsc ? ' ▲' : ' ▼';
    }
  });
}

function setupSearch() {
  document.getElementById('search').addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#op-table-body tr');
    rows.forEach(row => {
      const text = row.textContent.toLowerCase();
      row.style.display = text.includes(q) ? '' : 'none';
    });
  });
}

function setupSort() {
  document.querySelectorAll('#op-table th').forEach(th => {
    th.addEventListener('click', () => {
      const field = th.dataset.sort;
      if (!field) return;
      if (sortField === field) {
        sortAsc = !sortAsc;
      } else {
        sortField = field;
        sortAsc = true;
      }
      renderTable(dashboardData);
    });
  });
}

function showDetail(opName) {
  const op = dashboardData.operators.find(o => o.name === opName);
  if (!op) return;
  currentOp = op;
  currentTab = 'correctness';

  document.getElementById('detail-view').classList.add('visible');
  document.getElementById('detail-title').textContent = op.name + ' — Operator Detail';
  renderBasicInfo(op);
  renderDevStatus(op);
  renderCorrectness(op);
  renderPerformance(op);
  renderComparison(op);
  renderKernelTimeline(op);
  renderKernelTypeChart(op);
  renderProfiler(op);
  renderHistory(op);

  // Activate default tab
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="correctness"]').classList.add('active');
  document.getElementById('tab-correctness').classList.add('active');

  document.getElementById('detail-view').scrollIntoView({ behavior: 'smooth' });
}

function closeDetail() {
  document.getElementById('detail-view').classList.remove('visible');
  currentOp = null;
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
  document.getElementById(`tab-${tab}`).classList.add('active');
}

function kernelTag(type) {
  if (!type) return '<span class="kernel-tag default">N/A</span>';
  const t = type.toUpperCase();
  if (t.includes('AIVEC')) return '<span class="kernel-tag aivec">AIVEC</span>';
  if (t.includes('MIX_AIC') || t.includes('MIX') || t === 'MIX_AIC') return '<span class="kernel-tag mix">MIX_AIC</span>';
  if (t.includes('AICPU')) return '<span class="kernel-tag aicpu">AICPU</span>';
  if (t.includes('AIC')) return '<span class="kernel-tag aic">AIC</span>';
  return `<span class="kernel-tag default">${type}</span>`;
}

function renderBasicInfo(op) {
  document.getElementById('info-formula').textContent = op.formula || 'N/A';
  document.getElementById('info-shape').textContent = op.shape || 'N/A';
  document.getElementById('info-dtype').textContent = op.dtype || 'N/A';
  document.getElementById('info-batches').textContent = (op.batches || []).join(', ') || 'N/A';
  document.getElementById('info-broadcast').textContent = op.broadcast ? 'Yes' : 'No';
  document.getElementById('info-kernel-shape').textContent = op.kernel_shape || 'N/A';
  document.getElementById('info-logical-shape').textContent = op.logical_shape || 'N/A';
  document.getElementById('info-precision').textContent = op.precision || 'N/A';
  document.getElementById('info-fastest').textContent = op.fastest || 'N/A';
  document.getElementById('info-status').innerHTML = `<span class="status-badge ${op.status}">${op.status}</span>`;
}

function renderDevStatus(op) {
  const sd = op.dev_status_detail || {};
  const steps = ['Intent', 'API', 'Golden', 'Design', 'Implementation', 'Correctness', 'Benchmark', 'Archive'];
  const html = steps.map(s => {
    const st = sd[s] || 'pending';
    return `<span class="step ${st}">${st === 'completed' ? '✓' : st === 'failed' ? '✗' : '○'} ${s}</span>`;
  }).join('<span class="arrow">→</span>');
  document.getElementById('dev-pipeline').innerHTML = html;
}

function renderCorrectness(op) {
  const corr = op.correctness || {};
  const allPass = op.correctness_all_pass;
  const statusHtml = allPass === true ? '<span class="status-badge pass" style="font-size:16px;padding:4px 16px">ALL PASS</span>' :
                     allPass === false ? '<span class="status-badge fail" style="font-size:16px;padding:4px 16px">FAIL</span>' :
                     '<span class="status-badge unknown" style="font-size:16px;padding:4px 16px">UNKNOWN</span>';
  document.getElementById('corr-status').innerHTML = statusHtml;

  // Per-batch per-impl table
  const batches = op.batches || [];
  if (batches.length === 0) {
    document.getElementById('corr-table-body').innerHTML = '<tr><td colspan="6">No correctness data</td></tr>';
    return;
  }

  const impls = ['torch', 'ascendc', 'pypto'];
  let html = '';
  for (const bk of batches) {
    const sk = String(bk);
    html += `<tr><td>B=${bk}</td>`;
    for (const impl of impls) {
      const c = corr[impl]?.[sk];
      const s = c?.status || 'N/A';
      const cls = s === 'PASS' ? 'pass' : s === 'FAIL' ? 'fail' : 'unknown';
      const md = c?.max_abs_diff != null ? Number(c.max_abs_diff).toExponential(2) : '-';
      html += `<td><span class="status-badge ${cls}">${s}</span></td>
               <td style="font-size:12px">md=${md}</td>`;
    }
    html += '</tr>';
  }
  document.getElementById('corr-table-body').innerHTML = html;

  // Heatmap
  const heatmapContainer = document.getElementById('corr-heatmap');
  let hm = '<div class="heatmap" style="grid-template-columns:repeat(' + impls.length + ', 28px)">';
  for (const impl of impls) {
    hm += `<div style="font-size:10px;color:var(--text-muted);text-align:center">${impl[0].toUpperCase() + impl.slice(1,3)}</div>`;
  }
  for (const bk of batches) {
    const sk = String(bk);
    for (const impl of impls) {
      const c = corr[impl]?.[sk];
      const s = c?.status || 'N/A';
      const cls = s === 'PASS' ? 'pass' : s === 'FAIL' ? 'fail' : 'na';
      hm += `<div class="heatmap-cell ${cls}">${s === 'PASS' ? '✓' : s === 'FAIL' ? '✗' : '?'}</div>`;
    }
  }
  hm += '</div>';
  heatmapContainer.innerHTML = hm;

  // Note
  document.getElementById('corr-note').textContent = corr.note || '';
}

function renderPerformance(op) {
  const pdata = op.profiler_data || {};
  const summary = op.comparison_summary || [];
  const batches = op.batches || [];
  const container = document.getElementById('perf-table-body');

  if (summary.length > 0) {
    let html = '';
    for (const s of summary) {
      const bk = s.batch;
      const t = s.torch_primary_us ?? s.torch_us ?? 'N/A';
      const a = s.ascendc_primary_us ?? s.ascendc_us ?? 'N/A';
      const pc = s.pypto_primary_us ?? s.pypto_compute_us ?? 'N/A';
      const pt = s.pypto_total_dev_us ?? s.pypto_total_us ?? 'N/A';
      const f = s.fastest || '';
      const tStr = t !== 'N/A' ? Number(t).toFixed(1) : 'N/A';
      const aStr = a !== 'N/A' ? Number(a).toFixed(1) : 'N/A';
      const pcStr = pc !== 'N/A' ? Number(pc).toFixed(1) : 'N/A';
      const ptStr = pt !== 'N/A' ? Number(pt).toFixed(1) : 'N/A';
      html += `<tr>
        <td>B=${bk}</td>
        <td class="${f === 'torch' ? 'fastest' : ''}">${tStr} μs</td>
        <td class="${f === 'ascendc' ? 'fastest' : ''}">${aStr} μs</td>
        <td class="${f === 'pypto' ? 'fastest' : ''}">${pcStr} μs</td>
        <td>${ptStr} μs</td>
      </tr>`;
    }
    container.innerHTML = html;
  } else if (Object.keys(pdata).length > 0) {
    let html = '';
    for (const [bk, bv] of Object.entries(pdata)) {
      const t = bv.torch?.primary_compute_kernel_us;
      const a = bv.ascendc?.primary_compute_kernel_us;
      const p = bv.pypto?.primary_compute_kernel_us;
      const pt = bv.pypto?.all_device_kernels_per_call_us ?? bv.pypto?.all_device_kernels_us;
      html += `<tr>
        <td>B=${bk}</td>
        <td>${t != null ? t.toFixed(1) + ' μs' : 'N/A'}</td>
        <td>${a != null ? a.toFixed(1) + ' μs' : 'N/A'}</td>
        <td>${p != null ? p.toFixed(1) + ' μs' : 'N/A'}</td>
        <td>${pt != null ? pt.toFixed(1) + ' μs' : 'N/A'}</td>
      </tr>`;
    }
    container.innerHTML = html;
  } else {
    // Try benchmark data
    const impls = ['torch', 'ascendc', 'pypto'];
    let html = '';
    for (const bk of batches) {
      const sk = String(bk);
      html += `<tr><td>B=${bk}</td>`;
      for (const impl of impls) {
        const bm = op[impl]?.benchmark?.[sk];
        const v = bm?.median_us;
        html += `<td>${v != null ? v.toFixed(1) + ' μs' : 'N/A'}</td>`;
      }
      html += '</tr>';
    }
    container.innerHTML = html || '<tr><td colspan="5">No performance data</td></tr>';
  }
}

function renderComparison(op) {
  const summary = op.comparison_summary || [];
  const pdata = op.profiler_data || {};
  const container = document.getElementById('comparison-body');
  const batches = op.batches || [];

  if (summary.length > 0) {
    const ref = summary[0];
    cellL = document.getElementById('comp-ktype-torch');
    cellL.textContent = ref.torch_kernel_type || 'N/A';
    cellL = document.getElementById('comp-ktype-ascendc');
    cellL.textContent = ref.ascendc_kernel_type || 'N/A';
    cellL = document.getElementById('comp-ktype-pypto');
    cellL.textContent = ref.pypto_primary_type || 'N/A';

    const kc = op.kernel_count || {};
    document.getElementById('comp-kcount-torch').textContent = kc.torch ?? 'N/A';
    document.getElementById('comp-kcount-ascendc').textContent = kc.ascendc ?? 'N/A';
    document.getElementById('comp-kcount-pypto').textContent = kc.pypto ?? 'N/A';

    const latencies = ['torch_primary_us', 'ascendc_primary_us', 'pypto_primary_us'].map(k => {
      const v = ref[k] || ref[k.replace('_primary_us', '_us')];
      return v != null ? Number(v).toFixed(1) + ' μs' : 'N/A';
    });
    document.getElementById('comp-latency-torch').textContent = latencies[0];
    document.getElementById('comp-latency-ascendc').textContent = latencies[1];
    document.getElementById('comp-latency-pypto').textContent = latencies[2];

    // BW, GFLOPS from summary
    const bw = [ref.torch_bw_gbs, ref.ascendc_bw_gbs, ref.pypto_bw_gbs];
    document.getElementById('comp-bw-torch').textContent = bw[0] != null ? bw[0].toFixed(1) + ' GB/s' : 'N/A';
    document.getElementById('comp-bw-ascendc').textContent = bw[1] != null ? bw[1].toFixed(1) + ' GB/s' : 'N/A';
    document.getElementById('comp-bw-pypto').textContent = bw[2] != null ? bw[2].toFixed(1) + ' GB/s' : 'N/A';
  } else {
    // Try to get from pdata
    const firstKey = Object.keys(pdata)[0];
    if (firstKey) {
      const bv = pdata[firstKey];
      document.getElementById('comp-ktype-torch').textContent = bv.torch?.kernel_type || 'N/A';
      document.getElementById('comp-ktype-ascendc').textContent = bv.ascendc?.kernel_type || 'N/A';
      document.getElementById('comp-ktype-pypto').textContent = bv.pypto?.primary_kernel_type || bv.pypto?.kernel_type || 'N/A';

      const kc = op.kernel_count || {};
      document.getElementById('comp-kcount-torch').textContent = kc.torch ?? bv.torch?.kernels_per_call ?? 'N/A';
      document.getElementById('comp-kcount-ascendc').textContent = kc.ascendc ?? bv.ascendc?.kernels_per_call ?? 'N/A';
      document.getElementById('comp-kcount-pypto').textContent = kc.pypto ?? 'N/A';

      const getLat = (d) => d?.primary_compute_kernel_us != null ? d.primary_compute_kernel_us.toFixed(1) + ' μs' : 'N/A';
      document.getElementById('comp-latency-torch').textContent = getLat(bv.torch);
      document.getElementById('comp-latency-ascendc').textContent = getLat(bv.ascendc);
      document.getElementById('comp-latency-pypto').textContent = getLat(bv.pypto);
    }
  }

  // Correctness
  const allPass = op.correctness_all_pass;
  const corrStatus = allPass === true ? 'PASS' : allPass === false ? 'FAIL' : 'UNKNOWN';
  document.getElementById('comp-corr-torch').textContent = corrStatus;
  document.getElementById('comp-corr-ascendc').textContent = corrStatus;
  document.getElementById('comp-corr-pypto').textContent = corrStatus;
}

function renderKernelTimeline(op) {
  const pdata = op.profiler_data || {};
  const firstKey = Object.keys(pdata)[0];
  const container = document.getElementById('kernel-timeline');

  if (!firstKey) {
    container.innerHTML = '<div class="section"><h3>Kernel Timeline</h3><p class="text-secondary">No profiler data</p></div>';
    return;
  }

  const bv = pdata[firstKey];
  const tk = bv.torch?.primary_compute_kernel_us || 0;
  const ak = bv.ascendc?.primary_compute_kernel_us || 0;
  const pk = bv.pypto?.primary_compute_kernel_us || 0;
  const pt = bv.pypto?.all_device_kernels_per_call_us || bv.pypto?.all_device_kernels_us || 0;

  const maxDur = Math.max(tk, ak, pt, 1);
  const scale = 400 / maxDur;

  const tBar = (tk * scale).toFixed(0);
  const aBar = (ak * scale).toFixed(0);
  const pBar = (pk * scale).toFixed(0);
  const ptBar = (pt * scale).toFixed(0);

  const tn = bv.torch?.kernel_names?.[0] || 'torch_kernel';
  const an = bv.ascendc?.kernel_names?.[0] || 'ascendc_kernel';
  const pn = bv.pypto?.kernel_names?.[0] || 'pypto_kernel';

  container.innerHTML = `
    <div class="section">
      <h3>Kernel Timeline <span style="font-size:12px;color:var(--text-muted)">(B=${firstKey})</span></h3>
      <div class="kernel-timeline">
        <div class="kernel-row">
          <div class="kernel-label">Torch</div>
          <div class="kernel-bar-wrapper">
            <div class="kernel-bar torch" style="width:${tBar}px"><span class="bar-label">${tn}</span></div>
          </div>
          <div class="kernel-dur">${tk.toFixed(1)} μs</div>
        </div>
        <div class="kernel-row">
          <div class="kernel-label">Ascend C</div>
          <div class="kernel-bar-wrapper">
            <div class="kernel-bar ascendc" style="width:${aBar}px"><span class="bar-label">${an}</span></div>
          </div>
          <div class="kernel-dur">${ak.toFixed(1)} μs</div>
        </div>
        <div class="kernel-row">
          <div class="kernel-label">PyPTO<br><span style="font-size:11px;color:var(--text-muted)">compute</span></div>
          <div class="kernel-bar-wrapper">
            <div class="kernel-bar pypto" style="width:${pBar}px"><span class="bar-label">${pn}</span></div>
          </div>
          <div class="kernel-dur">${pk.toFixed(1)} μs</div>
        </div>
        <div class="kernel-row">
          <div class="kernel-label">PyPTO<br><span style="font-size:11px;color:var(--text-muted)">total</span></div>
          <div class="kernel-bar-wrapper">
            <div class="kernel-bar pypto" style="width:${ptBar}px"><span class="bar-label">+ executor</span></div>
          </div>
          <div class="kernel-dur">${pt.toFixed(1)} μs</div>
        </div>
      </div>
    </div>`;
}

function renderKernelTypeChart(op) {
  const container = document.getElementById('kernel-type-chart');
  const types = op.kernel_types || {};
  const count = op.kernel_count || {};

  const allTypes = {};
  for (const [impl, kts] of Object.entries(types)) {
    for (const kt of (Array.isArray(kts) ? kts : [kts])) {
      if (kt && kt !== 'N/A') {
        allTypes[kt] = (allTypes[kt] || 0) + 1;
      }
    }
  }

  if (Object.keys(allTypes).length === 0) {
    container.innerHTML = '<p class="text-secondary">No kernel type data</p>';
    return;
  }

  const colors = {
    'KERNEL_AIVEC': '#58a6ff',
    'KERNEL_MIX_AIC': '#bc8cff',
    'KERNEL_AICPU': '#f0883e',
    'KERNEL_AIC': '#39d2c0',
  };

  const total = Object.values(allTypes).reduce((a, b) => a + b, 0);
  let html = '<div style="display:flex;align-items:center;gap:24px"><div style="position:relative;width:200px;height:200px">';
  html += '<canvas id="kernel-pie" width="200" height="200"></canvas></div><div>';

  // Legend
  html += '<div class="legend">';
  for (const [kt, cnt] of Object.entries(allTypes)) {
    const pct = Math.round(cnt / total * 100);
    const c = colors[kt] || '#6e7681';
    html += `<div class="legend-item"><span class="dot" style="background:${c}"></span>${kt}: ${cnt} (${pct}%)</div>`;
  }
  html += '</div></div></div>';
  container.innerHTML = html;

  // Draw pie
  setTimeout(() => {
    const canvas = document.getElementById('kernel-pie');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const entries = Object.entries(allTypes);
    let startAngle = -Math.PI / 2;
    entries.forEach(([kt, cnt]) => {
      const sliceAngle = (cnt / total) * 2 * Math.PI;
      const c = colors[kt] || '#6e7681';
      ctx.beginPath();
      ctx.moveTo(100, 100);
      ctx.arc(100, 100, 90, startAngle, startAngle + sliceAngle);
      ctx.closePath();
      ctx.fillStyle = c;
      ctx.fill();
      startAngle += sliceAngle;
    });
  }, 50);
}

function renderProfiler(op) {
  const parsed = op.parsed || {};
  const container = document.getElementById('profiler-body');

  const entries = Object.entries(parsed).slice(0, 5);
  if (entries.length === 0) {
    container.innerHTML = '<tr><td colspan="7">No profiler data</td></tr>';
    return;
  }

  let html = '';
  for (const [key, pdata] of entries) {
    const events = pdata.kernel?.kernel_events || [];
    const byType = pdata.kernel?.by_type || {};
    const byName = pdata.kernel?.by_name || {};
    const totalDur = pdata.kernel?.total_kernel_dur_us || 0;
    const kernelCount = pdata.kernel?.kernel_count || 0;

    for (const [name, info] of Object.entries(byName)) {
      html += `<tr>
        <td style="font-size:12px">${key}</td>
        <td style="font-size:12px">${name}</td>
        <td>${info.mean_dur_us != null ? info.mean_dur_us.toFixed(2) + ' μs' : 'N/A'}</td>
        <td>${info.count}</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>N/A</td>
      </tr>`;
    }

    for (const [type, info] of Object.entries(byType)) {
      html += `<tr>
        <td style="font-size:12px">${key}</td>
        <td style="font-size:12px"><span class="kernel-tag ${type.toLowerCase().includes('aivec') ? 'aivec' : type.toLowerCase().includes('aicpu') ? 'aicpu' : type.toLowerCase().includes('mix') ? 'mix' : 'default'}">${type}</span></td>
        <td>${info.mean_dur_us != null ? info.mean_dur_us.toFixed(2) + ' μs' : 'N/A'}</td>
        <td>${info.count}</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>N/A</td>
      </tr>`;
    }
  }
  container.innerHTML = html;
}

function renderHistory(op) {
  const history = op.history || [];
  const container = document.getElementById('history-content');

  if (history.length === 0) {
    container.innerHTML = '<p class="text-secondary">No history data</p>';
    return;
  }

  let html = '';
  for (const h of history) {
    const hd = h.data;
    const corr = hd.correctness?.all_batches_pass !== undefined ? (hd.correctness.all_batches_pass ? 'PASS' : 'FAIL') : 'N/A';
    const perfStr = hd.comparison_summary?.[0] ? 'B1 torch=' + hd.comparison_summary[0].torch_primary_us?.toFixed(1) + 'μs' : 'N/A';
    html += `<div class="history-item">
      <div class="version">${h.version}</div>
      <div style="display:flex;gap:24px;margin-top:4px;font-size:13px">
        <span>Correctness: <span class="status-badge ${corr === 'PASS' ? 'pass' : 'fail'}">${corr}</span></span>
        <span>Latency: ${perfStr}</span>
      </div>
    </div>`;
  }
  container.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', init);
"""


def generate_html(dashboard):
    s = dashboard["summary"]
    ops_json = json.dumps(dashboard, ensure_ascii=False)
    css = generate_css()
    js = generate_js()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PyPTO Operator Dashboard</title>
<style>{css}</style>
</head>
<body>
<div id="loading" style="display:flex;align-items:center;justify-content:center;min-height:100vh;color:var(--text-muted);font-size:18px">
  Loading dashboard...
</div>

<div id="app" style="display:none">
  <div class="header">
    <div>
      <h1>PyPTO Operator Dashboard</h1>
      <div class="subtitle">Ascend C vs PyPTO — Development & Performance Tracking</div>
    </div>
    <div class="header-right">
      <span class="badge">{s["completed"]}/{s["total"]} completed</span>
      <span class="update-time" id="update-time"></span>
    </div>
  </div>

  <div class="toolbar">
    <input type="text" id="search" placeholder="Search operators...">
    <label style="color:var(--text-muted);font-size:13px">Sort by: click table headers</label>
  </div>

  <div class="container">
    <!-- Summary Cards -->
    <div class="summary-cards" id="summary-cards"></div>

    <!-- Progress Bar -->
    <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:24px">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:13px;color:var(--text-secondary)">Overall Completion</span>
        <span style="font-size:13px;font-weight:600" id="progress-text"></span>
      </div>
      <div class="progress-bar">
        <div class="fill" id="progress-fill" style="background:var(--accent-green)"></div>
      </div>
    </div>

    <!-- Operator Table -->
    <table id="op-table">
      <thead>
        <tr>
          <th data-sort="name">Operator <span class="sort-arrow">▲</span></th>
          <th data-sort="status">Status <span class="sort-arrow"></span></th>
          <th>Torch</th>
          <th>Ascend C</th>
          <th>PyPTO</th>
          <th>Correctness</th>
          <th>Performance (B=1)</th>
          <th>Last Update</th>
        </tr>
      </thead>
      <tbody id="op-table-body"></tbody>
    </table>

    <!-- Detail View -->
    <div class="detail-view" id="detail-view">
      <div class="detail-header">
        <h2 id="detail-title">Operator Detail</h2>
        <button class="detail-close" onclick="closeDetail()">Close</button>
      </div>

      <!-- Basic Info -->
      <div class="info-grid">
        <div class="info-item"><div class="label">Formula</div><div class="value" id="info-formula">-</div></div>
        <div class="info-item"><div class="label">Shape</div><div class="value" id="info-shape">-</div></div>
        <div class="info-item"><div class="label">Dtype</div><div class="value" id="info-dtype">-</div></div>
        <div class="info-item"><div class="label">Batches</div><div class="value" id="info-batches">-</div></div>
        <div class="info-item"><div class="label">Broadcast</div><div class="value" id="info-broadcast">-</div></div>
        <div class="info-item"><div class="label">Kernel Shape</div><div class="value" id="info-kernel-shape">-</div></div>
        <div class="info-item"><div class="label">Logical Shape</div><div class="value" id="info-logical-shape">-</div></div>
        <div class="info-item"><div class="label">Precision</div><div class="value" id="info-precision">-</div></div>
        <div class="info-item"><div class="label">Fastest</div><div class="value" id="info-fastest">-</div></div>
        <div class="info-item"><div class="label">Status</div><div class="value" id="info-status">-</div></div>
      </div>

      <!-- Dev Status Pipeline -->
      <div class="section">
        <h3>Development Status</h3>
        <div class="pipeline" id="dev-pipeline"></div>
      </div>

      <!-- Tabs -->
      <div class="tabs">
        <div class="tab active" data-tab="correctness" onclick="switchTab('correctness')">Correctness</div>
        <div class="tab" data-tab="performance" onclick="switchTab('performance')">Performance</div>
        <div class="tab" data-tab="comparison" onclick="switchTab('comparison')">Comparison</div>
        <div class="tab" data-tab="kernel" onclick="switchTab('kernel')">Kernel Timeline</div>
        <div class="tab" data-tab="ktype" onclick="switchTab('ktype')">Kernel Type</div>
        <div class="tab" data-tab="profiler" onclick="switchTab('profiler')">Profiler</div>
        <div class="tab" data-tab="history" onclick="switchTab('history')">History</div>
      </div>

      <!-- Tab: Correctness -->
      <div class="tab-content active" id="tab-correctness">
        <div class="section">
          <h3>Correctness Summary</h3>
          <div id="corr-status" style="margin-bottom:12px"></div>
          <div id="corr-note" style="font-size:13px;color:var(--text-secondary);margin-bottom:12px"></div>
        </div>
        <div class="section">
          <h3>Per-Batch Results</h3>
          <table>
            <thead><tr><th>Batch</th><th>Torch</th><th>Diff</th><th>AscendC</th><th>Diff</th><th>PyPTO</th><th>Diff</th></tr></thead>
            <tbody id="corr-table-body"></tbody>
          </table>
        </div>
        <div class="section">
          <h3>Correctness Heatmap</h3>
          <div id="corr-heatmap"></div>
        </div>
      </div>

      <!-- Tab: Performance -->
      <div class="tab-content" id="tab-performance">
        <div class="section">
          <h3>Latency Comparison (μs)</h3>
          <table>
            <thead><tr><th>Batch</th><th>Torch</th><th>Ascend C</th><th>PyPTO Compute</th><th>PyPTO Total</th></tr></thead>
            <tbody id="perf-table-body"></tbody>
          </table>
        </div>
      </div>

      <!-- Tab: Comparison -->
      <div class="tab-content" id="tab-comparison">
        <div class="section">
          <h3>Side-by-Side Comparison</h3>
          <table class="comparison-table">
            <thead><tr><th>Metric</th><th>Torch</th><th>Ascend C</th><th>PyPTO</th></tr></thead>
            <tbody id="comparison-body">
              <tr><td>Kernel Type</td><td id="comp-ktype-torch">-</td><td id="comp-ktype-ascendc">-</td><td id="comp-ktype-pypto">-</td></tr>
              <tr><td>Kernel Count</td><td id="comp-kcount-torch">-</td><td id="comp-kcount-ascendc">-</td><td id="comp-kcount-pypto">-</td></tr>
              <tr><td>Latency (B=1)</td><td id="comp-latency-torch">-</td><td id="comp-latency-ascendc">-</td><td id="comp-latency-pypto">-</td></tr>
              <tr><td>Bandwidth</td><td id="comp-bw-torch">-</td><td id="comp-bw-ascendc">-</td><td id="comp-bw-pypto">-</td></tr>
              <tr><td>Correctness</td><td id="comp-corr-torch">-</td><td id="comp-corr-ascendc">-</td><td id="comp-corr-pypto">-</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Tab: Kernel Timeline -->
      <div class="tab-content" id="tab-kernel">
        <div id="kernel-timeline"></div>
      </div>

      <!-- Tab: Kernel Type -->
      <div class="tab-content" id="tab-ktype">
        <div class="section">
          <h3>Kernel Type Distribution</h3>
          <div id="kernel-type-chart"></div>
        </div>
      </div>

      <!-- Tab: Profiler -->
      <div class="tab-content" id="tab-profiler">
        <div class="section">
          <h3>Profiler Data</h3>
          <table>
            <thead><tr><th>Run</th><th>Kernel Name</th><th>Duration</th><th>Calls</th><th>Core Type</th><th>BlockDim</th><th>Tile</th></tr></thead>
            <tbody id="profiler-body"></tbody>
          </table>
        </div>
      </div>

      <!-- Tab: History -->
      <div class="tab-content" id="tab-history">
        <div class="section">
          <h3>Version History</h3>
          <div id="history-content"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>{js}</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
