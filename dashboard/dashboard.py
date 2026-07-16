#!/usr/bin/env python3
"""PyPTO Operator Dashboard.

Development mode:
    python dashboard.py

    Scans operators/*/ for live data. Generates dashboard/index.html + dashboard/dashboard.json.

Release mode:
    python dashboard.py --release reports/release/current_release.json

    Reads only the release JSON — does NOT scan operators/. Generates dashboard/index.html + dashboard/dashboard.json.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OPERATORS = BASE / "operators"
OUT = BASE / "dashboard"

# ---------------------------------------------------------------------------
# Release-mode data loader
# ---------------------------------------------------------------------------

def load_release(path):
    """Load release JSON and convert to dashboard format.

    Everything must come from the release JSON — no operator directory scanning.
    """
    raw = json.loads(Path(path).read_text())
    ops = {}
    for name, op in raw.get("operators", {}).items():
        routes = op.get("routes", {})
        torch_r = routes.get("torch", {})
        ascendc_r = routes.get("ascendc", {})
        pypto_r = routes.get("pypto", {})

        # Correctness
        correctness = {}
        for route_key, label in [("torch", "torch"), ("ascendc", "ascendc"), ("pypto", "pypto")]:
            r = routes.get(route_key, {})
            c = r.get("correctness", "N/A")
            correctness[label] = c

        # Profiler summary
        profiler = {}
        for route_key, label in [("torch", "torch"), ("ascendc", "ascendc"), ("pypto", "pypto")]:
            p = r.get("profiler", {})
            pstat = {
                "status": p.get("status", "N/A"),
                "method": p.get("method", "N/A"),
            }
            if p.get("not_comparable"):
                pstat["not_comparable"] = True
            if p.get("b1_primary_us") is not None:
                pstat["b1_us"] = p["b1_primary_us"]
            elif p.get("b1_host_sync_us") is not None:
                pstat["b1_us"] = p["b1_host_sync_us"]
            profiler[label] = pstat

        # Profiler tool detection from batches
        profiler_tool = "NONE"
        for batch_key, batch in raw.get("batches", {}).items():
            if op.get("shape", "") and any(
                kw in op.get("shape", "").lower() for kw in batch.get("operators", [name])
            ):
                pass
            if name in batch.get("operators", []):
                p = batch.get("profiler", "none")
                profiler_tool = p if p != "none" else "NONE"
                break
        else:
            profiler_tool = "NONE"

        ops[name] = {
            "status": op.get("final_status", "UNKNOWN"),
            "formula": op.get("formula", ""),
            "shape": op.get("shape", ""),
            "dtype": op.get("dtype", ""),
            "batches": op.get("batches", []),
            "precision": op.get("precision", ""),
            "limitation": op.get("limitation", ""),
            "correctness": correctness,
            "profiler": profiler,
            "profiler_tool": profiler_tool,
            "report_path": op.get("report_path", ""),
            "archive": op.get("archive", "none"),
            "completeness_gates": op.get("completeness_gates", []),
        }

    # Compute summary
    total = len(ops)
    status_counts = {}
    for op in ops.values():
        s = op["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    # Batch group info
    batches_info = {}
    for bk, bv in raw.get("batches", {}).items():
        batches_info[bk] = {
            "operators": bv.get("operators", []),
            "profiler": bv.get("profiler", "none"),
            "not_comparable": bv.get("not_comparable_with_arithmetic", False),
        }

    return {
        "mode": "release",
        "release_version": raw.get("release_version", "unknown"),
        "generated_at": raw.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "environment": raw.get("environment", {}),
        "operator_count": total,
        "status_summary": status_counts,
        "batches": batches_info,
        "operators": ops,
        "known_limitations": raw.get("known_limitations", []),
        "ascendc_implementation_audit": raw.get("ascendc_implementation_audit", {}),
        "archive_references": {k: v["archive"] for k, v in ops.items()},
    }


# ---------------------------------------------------------------------------
# Development-mode data loader (preserved from original)
# ---------------------------------------------------------------------------

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
        if data["correctness_all_pass"] is None:
            for impl_name in ("torch", "ascendc", "pypto"):
                c = final.get("correctness", {}).get(impl_name, {})
                if isinstance(c, dict):
                    st = c.get("status", "")
                    if st == "PASS":
                        data["correctness_all_pass"] = True
                    elif st == "FAIL":
                        data["correctness_all_pass"] = False

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

    # parsed reports
    parsed_dir = d / "reports" / "parsed"
    if parsed_dir.exists():
        parsed_files = sorted(parsed_dir.glob("*.json"))
        data["parsed"] = {}
        for pf in parsed_files:
            pdata_parsed = load_json(pf)
            if pdata_parsed:
                stem = pf.stem
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


def build_dev():
    ops_dirs = sorted([
        d.name for d in OPERATORS.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
    ])

    operators = []
    for name in ops_dirs:
        operators.append(get(name))

    total = len(operators)
    completed = sum(1 for o in operators if o["status"] == "completed")
    torch_done = sum(1 for o in operators if "KERNEL_AIVEC" in str(o["kernel_types"].get("torch", [])))
    ascendc_done = sum(1 for o in operators if "KERNEL_AIVEC" in str(o["kernel_types"].get("ascendc", [])))
    pypto_done = sum(1 for o in operators if o["pypto"])
    pass_count = sum(1 for o in operators if o["correctness_all_pass"] is True)
    fail_count = sum(1 for o in operators if o["correctness_all_pass"] is False)
    blocked = sum(1 for o in operators if o["status"] == "blocked")

    dashboard = {
        "mode": "development",
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
    return dashboard


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(dashboard):
    s = dashboard.get("summary", {})
    mode = dashboard.get("mode", "development")
    total = s.get("total", dashboard.get("operator_count", 0))
    completed = s.get("completed", 0)

    # Release mode source info
    release_info = ""
    if mode == "release":
        release_info = f'<p>Release v{dashboard.get("release_version", "?")} — {dashboard["generated_at"]}</p>'

    ops_json = json.dumps(dashboard, ensure_ascii=False)
    css = generate_css()
    js = generate_js()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PyPTO Operator Dashboard{f' v{dashboard.get("release_version", "")}' if mode == 'release' else ''}</title>
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
      <div class="subtitle">{'Release Mode — Source: reports/release/current_release.json' if mode == 'release' else 'Development Mode — Scanning operators/*/'}</div>
    </div>
    <div class="header-right">
      <span class="badge">{completed}/{total} completed</span>
      <span class="update-time" id="update-time"></span>
    </div>
  </div>

  <div class="toolbar">
    <input type="text" id="search" placeholder="Search operators...">
    <label style="color:var(--text-muted);font-size:13px">Sort by: click table headers</label>
  </div>

  <div class="container">
    {release_info}

    <div class="summary-cards" id="summary-cards"></div>

    <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:24px">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:13px;color:var(--text-secondary)">Overall Completion</span>
        <span style="font-size:13px;font-weight:600" id="progress-text"></span>
      </div>
      <div class="progress-bar">
        <div class="fill" id="progress-fill" style="background:var(--accent-green)"></div>
      </div>
    </div>

    <table id="op-table">
      <thead>
        <tr>
          <th data-sort="name">Operator <span class="sort-arrow">▲</span></th>
          <th data-sort="status">Status <span class="sort-arrow"></span></th>
          <th>Torch</th>
          <th>Ascend C</th>
          <th>PyPTO</th>
          <th>Correctness</th>
          {"<th>B1 Perf</th>" if mode == 'release' else '<th>Performance (B=1)</th>'}
          {"<th>Profiler</th>" if mode == 'release' else '<th>Last Update</th>'}
        </tr>
      </thead>
      <tbody id="op-table-body"></tbody>
    </table>

    <div class="detail-view" id="detail-view">
      <div class="detail-header">
        <h2 id="detail-title">Operator Detail</h2>
        <button class="detail-close" onclick="closeDetail()">Close</button>
      </div>

      <div class="info-grid">
        <div class="info-item"><div class="label">Formula</div><div class="value" id="info-formula">-</div></div>
        <div class="info-item"><div class="label">Shape</div><div class="value" id="info-shape">-</div></div>
        <div class="info-item"><div class="label">Dtype</div><div class="value" id="info-dtype">-</div></div>
        <div class="info-item"><div class="label">Batches</div><div class="value" id="info-batches">-</div></div>
        <div class="info-item"><div class="label">Precision</div><div class="value" id="info-precision">-</div></div>
        <div class="info-item"><div class="label">Status</div><div class="value" id="info-status">-</div></div>
        <div class="info-item"><div class="label">Limitation</div><div class="value" id="info-limitation">-</div></div>
        <div class="info-item"><div class="label">Archive</div><div class="value" id="info-archive">-</div></div>
      </div>

      <div class="tabs">
        <div class="tab active" data-tab="correctness" onclick="switchTab('correctness')">Correctness</div>
        <div class="tab" data-tab="performance" onclick="switchTab('performance')">Performance</div>
        <div class="tab" data-tab="limitations" onclick="switchTab('limitations')">Limitations</div>
      </div>

      <div class="tab-content active" id="tab-correctness">
        <div class="section">
          <h3>Correctness Summary</h3>
          <div id="corr-status" style="margin-bottom:12px"></div>
        </div>
        <div class="section">
          <h3>Per-Route Results</h3>
          <table>
            <thead><tr><th>Route</th><th>Result</th><th>Tool</th></tr></thead>
            <tbody id="corr-table-body"></tbody>
          </table>
        </div>
      </div>

      <div class="tab-content" id="tab-performance">
        <div class="section">
          <h3>Profiler Metrics</h3>
          <table>
            <thead><tr><th>Route</th><th>Method</th><th>B1 Latency</th></tr></thead>
            <tbody id="perf-table-body"></tbody>
          </table>
        </div>
      </div>

      <div class="tab-content" id="tab-limitations">
        <div class="section">
          <h3>Known Limitations</h3>
          <div id="limitations-content"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>{js}</script>
</body>
</html>"""


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
.header h1 { font-size: 20px; font-weight: 600; }
.header .subtitle { color: var(--text-secondary); font-size: 13px; margin-top: 2px; }
.header-right { display: flex; align-items: center; gap: 16px; }
.header-right .update-time { color: var(--text-muted); font-size: 12px; }
.header-right .badge { background: var(--accent-blue); color: #fff; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }

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
.toolbar input:focus { border-color: var(--accent-blue); }

.container { max-width: 1400px; margin: 0 auto; padding: 24px; }

.summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
.summary-card { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }
.summary-card .value { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
.summary-card .label { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
.summary-card.color-blue .value { color: var(--accent-blue); }
.summary-card.color-green .value { color: var(--accent-green); }
.summary-card.color-red .value { color: var(--accent-red); }
.summary-card.color-yellow .value { color: var(--accent-yellow); }
.summary-card.color-purple .value { color: var(--accent-purple); }
.summary-card.color-orange .value { color: var(--accent-orange); }

.progress-bar { background: var(--bg-tertiary); border-radius: 6px; height: 8px; overflow: hidden; margin-top: 8px; }
.progress-bar .fill { height: 100%; border-radius: 6px; transition: width 0.3s; }

table { width: 100%; border-collapse: collapse; background: var(--bg-secondary); border-radius: 8px; overflow: hidden; margin-bottom: 24px; }
th { background: var(--bg-tertiary); padding: 10px 14px; text-align: left; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-secondary); cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { color: var(--text-primary); }
th .sort-arrow { margin-left: 4px; opacity: 0.5; }
th.sorted .sort-arrow { opacity: 1; }
td { padding: 10px 14px; border-top: 1px solid var(--border); font-size: 14px; }
tr { cursor: pointer; }
tr:hover { background: var(--bg-hover); }

.status-badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; }
.status-badge.completed { background: rgba(63,185,80,0.15); color: var(--accent-green); }
.status-badge.in_progress { background: rgba(210,153,34,0.15); color: var(--accent-yellow); }
.status-badge.planned { background: rgba(110,118,129,0.15); color: var(--text-muted); }
.status-badge.blocked { background: rgba(248,81,73,0.15); color: var(--accent-red); }
.status-badge.pass { background: rgba(63,185,80,0.15); color: var(--accent-green); }
.status-badge.fail { background: rgba(248,81,73,0.15); color: var(--accent-red); }
.status-badge.unknown { background: rgba(110,118,129,0.15); color: var(--text-muted); }

.detail-view { display: none; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 24px; margin-bottom: 24px; }
.detail-view.visible { display: block; }
.detail-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.detail-header h2 { font-size: 24px; }
.detail-close { background: none; border: 1px solid var(--border); color: var(--text-secondary); padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 14px; }
.detail-close:hover { background: var(--bg-hover); color: var(--text-primary); }

.info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }
.info-item { background: var(--bg-tertiary); border-radius: 6px; padding: 12px 16px; }
.info-item .label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.info-item .value { font-size: 15px; font-weight: 500; }

.section { margin-bottom: 24px; }
.section h3 { font-size: 16px; font-weight: 600; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }

.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
.tab { padding: 8px 16px; cursor: pointer; font-size: 13px; font-weight: 500; color: var(--text-secondary); border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab:hover { color: var(--text-primary); }
.tab.active { color: var(--accent-blue); border-bottom-color: var(--accent-blue); }
.tab-content { display: none; }
.tab-content.active { display: block; }

.limitation-item { background: var(--bg-tertiary); border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; }
.limitation-item .sev-p0 { color: var(--accent-red); font-weight: 600; }
.limitation-item .sev-p1 { color: var(--accent-yellow); font-weight: 600; }
.limitation-item .sev-p2 { color: var(--accent-orange); font-weight: 600; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.summary-card, .detail-view.visible { animation: fadeIn 0.3s ease-out; }

@media (max-width: 768px) {
  .summary-cards { grid-template-columns: repeat(2, 1fr); }
  .info-grid { grid-template-columns: repeat(2, 1fr); }
  .toolbar input { width: 160px; }
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
  if (data.mode === 'release') {
    const s = data.status_summary || {};
    const total = data.operator_count || 0;
    const cards = [
      { label: 'Total Operators', value: total, cls: 'color-blue' },
    ];
    for (const [status, count] of Object.entries(s)) {
      const cls = status.startsWith('COMPLETE') ? 'color-green' : status === 'PARTIAL' ? 'color-yellow' : 'color-red';
      cards.push({ label: status, value: count, cls: cls });
    }
    const container = document.getElementById('summary-cards');
    container.innerHTML = cards.map(c => `
      <div class="summary-card ${c.cls}">
        <div class="value">${c.value}</div>
        <div class="label">${c.label}</div>
      </div>
    `).join('');

    const compl = (s['COMPLETE'] || 0) + (s['COMPLETE_WITH_LIMITATION'] || 0);
    const pct = total > 0 ? Math.round(compl / total * 100) : 0;
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-text').textContent = pct + '% (' + compl + '/' + total + ')';
    document.getElementById('update-time').textContent = 'Release: ' + data.generated_at;
    return;
  }

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

  const pct = s.total > 0 ? Math.round(s.completed / s.total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent = pct + '% (' + s.completed + '/' + s.total + ')';
  document.getElementById('update-time').textContent = 'Generated: ' + data.generated_at;
}

function renderTable(data) {
  if (data.mode === 'release') {
    renderReleaseTable(data);
    return;
  }
  renderDevTable(data);
}

function renderReleaseTable(data) {
  const ops = Object.entries(data.operators).map(([name, op]) => ({ name: name, ...op }));
  ops.sort((a, b) => {
    const order = { 'COMPLETE': 0, 'COMPLETE_WITH_LIMITATION': 1, 'PARTIAL': 2, 'INCOMPLETE': 3 };
    let va = (order[a.status] ?? 99);
    let vb = (order[b.status] ?? 99);
    return sortAsc ? va - vb : vb - va;
  });

  const tbody = document.getElementById('op-table-body');
  tbody.innerHTML = ops.map(op => {
    const statusCls = op.status.startsWith('COMPLETE') ? 'completed' : op.status === 'PARTIAL' ? 'in_progress' : 'unknown';
    const torchC = op.correctness?.torch || 'N/A';
    const ascendcC = op.correctness?.ascendc || 'N/A';
    const pyptoC = op.correctness?.pypto || 'N/A';

    const corrStr = [torchC, ascendcC, pyptoC].every(c => c.startsWith('PASS'))
      ? '<span class="status-badge pass">PASS</span>'
      : [torchC, ascendcC, pyptoC].some(c => c.startsWith('FAIL'))
        ? '<span class="status-badge fail">FAIL</span>'
        : '<span class="status-badge unknown">MIXED</span>';

    const torchP = op.profiler?.torch || {};
    const ascendcP = op.profiler?.ascendc || {};
    const pyptoP = op.profiler?.pypto || {};
    const b1 = torchP.b1_us != null ? 'T:' + Number(torchP.b1_us).toFixed(1) + 'us' : '';
    const b1a = ascendcP.b1_us != null ? ' A:' + Number(ascendcP.b1_us).toFixed(1) + 'us' : '';
    const b1p = pyptoP.b1_us != null ? ' P:' + Number(pyptoP.b1_us).toFixed(1) + 'us' : '';
    const b1Str = (b1 + b1a + b1p) || 'N/A';

    const profilerTool = op.profiler_tool || 'N/A';

    return `<tr onclick="showDetail('${op.name}')">
      <td><strong>${op.name}</strong></td>
      <td><span class="status-badge ${statusCls}">${op.status}</span></td>
      <td style="font-size:12px">${torchC}</td>
      <td style="font-size:12px">${ascendcC}</td>
      <td style="font-size:12px">${pyptoC}</td>
      <td>${corrStr}</td>
      <td style="font-size:12px">${b1Str}</td>
      <td style="font-size:12px">${profilerTool}</td>
    </tr>`;
  }).join('');
}

function renderDevTable(data) {
  const ops = [...(data.operators || [])];
  ops.sort((a, b) => {
    let va = String(a[sortField] || '').toLowerCase();
    let vb = String(b[sortField] || '').toLowerCase();
    if (sortField === 'status') {
      const order = { completed: 0, in_progress: 1, planned: 2, blocked: 3 };
      const da = order[a.status] || 99;
      const db = order[b.status] || 99;
      return sortAsc ? da - db : db - da;
    }
    if (sortField === 'name') {
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
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
    return `<tr onclick="showDevDetail('${op.name}')">
      <td><strong>${op.name}</strong></td>
      <td><span class="status-badge ${op.status}">${op.status}</span></td>
      <td>${torchKt}</td>
      <td>${ascendcKt}</td>
      <td>${pyptoKt}</td>
      <td>${corr}</td>
      <td style="font-size:12px">${op.last_update}</td>
    </tr>`;
  }).join('');

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
  if (dashboardData.mode === 'release') {
    showReleaseDetail(opName);
  } else {
    showDevDetail(opName);
  }
}

function showReleaseDetail(opName) {
  const op = dashboardData.operators[opName];
  if (!op) return;
  currentOp = op;

  document.getElementById('detail-view').classList.add('visible');
  document.getElementById('detail-title').textContent = opName + ' — Operator Detail';

  document.getElementById('info-formula').textContent = op.formula || 'N/A';
  document.getElementById('info-shape').textContent = op.shape || 'N/A';
  document.getElementById('info-dtype').textContent = op.dtype || 'N/A';
  document.getElementById('info-batches').textContent = (op.batches || []).join(', ') || 'N/A';
  document.getElementById('info-precision').textContent = op.precision || 'N/A';
  const statusCls = op.status.startsWith('COMPLETE') ? 'completed' : op.status === 'PARTIAL' ? 'in_progress' : 'unknown';
  document.getElementById('info-status').innerHTML = '<span class="status-badge ' + statusCls + '">' + op.status + '</span>';
  document.getElementById('info-limitation').textContent = op.limitation || 'None';
  document.getElementById('info-archive').textContent = op.archive || 'none';

  renderReleaseCorrectness(op);
  renderReleasePerformance(op);

  if (dashboardData.known_limitations) {
    const opLimits = dashboardData.known_limitations.filter(l => l.operator === opName);
    const container = document.getElementById('limitations-content');
    if (opLimits.length === 0) {
      container.innerHTML = '<p style="color:var(--text-muted)">No known limitations for this operator.</p>';
    } else {
      container.innerHTML = opLimits.map(l => {
        const sevClass = 'sev-' + l.severity.toLowerCase();
        return '<div class="limitation-item"><span class="' + sevClass + '">[' + l.severity + ']</span> ' + l.route + ': ' + l.description + ' <span style="color:var(--text-muted);font-size:12px">(' + l.blocker_type + ')</span></div>';
      }).join('');
    }
  }

  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="correctness"]').classList.add('active');
  document.getElementById('tab-correctness').classList.add('active');
  document.getElementById('detail-view').scrollIntoView({ behavior: 'smooth' });
}

function renderReleaseCorrectness(op) {
  const corr = op.correctness || {};
  const allPass = Object.values(corr).every(c => String(c).startsWith('PASS'));
  const allFail = Object.values(corr).every(c => String(c).startsWith('FAIL'));
  const statusHtml = allPass ? '<span class="status-badge pass" style="font-size:16px;padding:4px 16px">ALL PASS</span>' :
                     allFail ? '<span class="status-badge fail" style="font-size:16px;padding:4px 16px">ALL FAIL</span>' :
                     '<span class="status-badge unknown" style="font-size:16px;padding:4px 16px">MIXED</span>';
  document.getElementById('corr-status').innerHTML = statusHtml;

  const impls = ['torch', 'ascendc', 'pypto'];
  let html = '';
  for (const impl of impls) {
    const c = corr[impl] || 'N/A';
    html += '<tr><td>' + impl + '</td><td>' + c + '</td><td style="font-size:12px">' + (op.profiler?.[impl]?.method || 'N/A') + '</td></tr>';
  }
  document.getElementById('corr-table-body').innerHTML = html;
}

function renderReleasePerformance(op) {
  const profiler = op.profiler || {};
  const impls = ['torch', 'ascendc', 'pypto'];
  let html = '';
  for (const impl of impls) {
    const p = profiler[impl] || {};
    const method = p.method || 'N/A';
    const lat = p.b1_us != null ? Number(p.b1_us).toFixed(1) + ' us' : 'N/A';
    html += '<tr><td>' + impl + '</td><td>' + method + '</td><td>' + lat + '</td></tr>';
  }
  document.getElementById('perf-table-body').innerHTML = html;
}

function closeDetail() {
  document.getElementById('detail-view').classList.remove('visible');
  currentOp = null;
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="' + tab + '"]').classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}

document.addEventListener('DOMContentLoaded', init);
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyPTO Operator Dashboard Generator")
    parser.add_argument("--release", type=str, default=None,
                        help="Release mode: path to reports/release/current_release.json")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.release:
        path = Path(args.release)
        if not path.exists():
            print(f"[ERR] Release file not found: {path}", file=sys.stderr)
            sys.exit(1)
        dashboard = load_release(path)
        total = dashboard.get("operator_count", 0)
        print(f"[OK] Loaded release ({dashboard.get('release_version', '?')}) — {total} operators")
    else:
        print("[INFO] Development mode — scanning operators/*/")
        dashboard = build_dev()

    # Write dashboard.json
    out_json = OUT / "dashboard.json"
    out_json.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False, default=str))
    print(f"[OK] Written {out_json}")

    # Generate index.html
    html = generate_html(dashboard)
    out_html = OUT / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] Written {out_html}")

    # Write CSS (standalone, also embedded in HTML)
    css = generate_css()
    (OUT / "dashboard.css").write_text(css, encoding="utf-8")

    print(f"\nDashboard ready: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
