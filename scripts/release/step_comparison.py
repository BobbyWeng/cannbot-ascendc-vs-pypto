import json
import csv
import sys
from datetime import datetime
from release_config import (
    OPERATORS, OPERATOR_DIR, FINAL_DIR, RELEASE_DIR,
    FINAL_COMPARISON, FINAL_COMPARISON_JSON, FINAL_COMPARISON_CSV,
    CURRENT_RELEASE, CURRENT_RELEASE_MD, PERFORMANCE_MATRIX,
    DEFAULT_ENV, RELEASE_VERSION,
)

def run(dry_run=False, force=False):
    if dry_run:
        print("[DRY-RUN] step_comparison: would generate final comparison files")
        return True

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    _generate_final_comparison()
    _generate_performance_csv()
    _generate_current_release_md()
    return True


def _load_current_release():
    if CURRENT_RELEASE.exists():
        return json.loads(CURRENT_RELEASE.read_text())
    return None


def _collect_operator_data():
    release = _load_current_release()
    if release and "operators" in release:
        return release

    ops = {}
    for op in OPERATORS:
        op_dir = OPERATOR_DIR / op
        final_json = op_dir / "reports" / "final" / "final_comparison.json"
        if final_json.exists():
            ops[op] = json.loads(final_json.read_text())
    return {"operators": ops}


def _generate_final_comparison():
    release = _load_current_release()
    if not release:
        _log("current_release.json not found, generating minimal comparison")
        release = {
            "release_version": RELEASE_VERSION,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "environment": DEFAULT_ENV,
            "operator_count": len(OPERATORS),
            "status_summary": {"COMPLETE": 0, "COMPLETE_WITH_LIMITATION": 0, "PARTIAL": 0, "INCOMPLETE": 0},
            "operators": {},
        }
        for op in OPERATORS:
            op_dir = OPERATOR_DIR / op
            final_json = op_dir / "reports" / "final" / "final_comparison.json"
            if final_json.exists():
                release["operators"][op] = {"final_status": "COMPLETE"}

    status_counts = {"COMPLETE": 0, "COMPLETE_WITH_LIMITATION": 0, "PARTIAL": 0, "INCOMPLETE": 0}
    for op_name, op_data in release.get("operators", {}).items():
        fs = op_data.get("final_status", "INCOMPLETE")
        if fs in status_counts:
            status_counts[fs] += 1

    comparison = {
        "release_version": release.get("release_version", RELEASE_VERSION),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": release.get("environment", DEFAULT_ENV),
        "operator_count": len(release.get("operators", {})),
        "status_summary": status_counts,
        "operators": {},
    }
    for op_name, op_data in release.get("operators", {}).items():
        routes = op_data.get("routes", {})
        comparison["operators"][op_name] = {
            "final_status": op_data.get("final_status", "UNKNOWN"),
            "formula": op_data.get("formula", ""),
            "shape": op_data.get("shape", ""),
            "dtype": op_data.get("dtype", ""),
            "batches": op_data.get("batches", []),
            "precision": op_data.get("precision", ""),
            "correctness_coverage": op_data.get("correctness_coverage", ""),
            "profiler_coverage": op_data.get("profiler_coverage", ""),
            "limitation": op_data.get("limitation", ""),
        }
        route_info = {}
        for route_key in ("torch", "ascendc", "pypto"):
            r = routes.get(route_key, {})
            route_info[route_key] = {
                "correctness": r.get("correctness", "N/A"),
                "profiler_method": r.get("profiler", {}).get("method", "N/A"),
            }
            p = r.get("profiler", {})
            if p.get("b1_primary_us") is not None:
                route_info[route_key]["b1_us"] = p["b1_primary_us"]
        comparison["operators"][op_name]["routes"] = route_info

    comparison["known_limitations"] = release.get("known_limitations", [])
    comparison["ascendc_implementation_audit"] = release.get("ascendc_implementation_audit", {})

    FINAL_COMPARISON_JSON.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    _log(f"Final comparison JSON -> {FINAL_COMPARISON_JSON}")

    _write_comparison_md(comparison)

    csv_path = _write_comparison_csv(comparison)
    _log(f"Final comparison CSV -> {csv_path}")
    return comparison


def _write_comparison_md(comparison):
    lines = [
        f"# Cannbot {comparison['release_version']}: Ascend C vs PyPTO — Final Comparison",
        "",
        "## Release Info",
        f"- **Version**: {comparison['release_version']}",
        f"- **Generated**: {comparison['generated_at']}",
        f"- **Environment**: {comparison['environment'].get('platform', 'N/A')}, CANN {comparison['environment'].get('cann_version', 'N/A')}",
        f"- **Operators**: {comparison['operator_count']}",
        "",
        "## Overall Status",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for status, count in comparison.get("status_summary", {}).items():
        lines.append(f"| {status} | {count} |")
    lines.append("")

    lines.append("## Operator Detail")
    lines.append("")
    for op_name, op_data in comparison.get("operators", {}).items():
        lines.append(f"### {op_name} — {op_data.get('final_status', 'UNKNOWN')}")
        lines.append(f"| Route | Correctness | B1 Latency |")
        lines.append(f"|-------|-------------|:----------:|")
        for route_key in ("torch", "ascendc", "pypto"):
            r = op_data.get("routes", {}).get(route_key, {})
            corr = r.get("correctness", "N/A")
            lat = f"{r.get('b1_us', 'N/A')} us" if r.get("b1_us") else "N/A"
            lines.append(f"| {route_key} | {corr} | {lat} |")
        if op_data.get("limitation"):
            lines.append(f"\n**Limitation**: {op_data['limitation']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Known Limitations")
    lines.append("")
    lines.append("| Operator | Route | Severity | Description |")
    lines.append("|----------|-------|:--------:|-------------|")
    for lim in comparison.get("known_limitations", []):
        lines.append(f"| {lim.get('operator', '?')} | {lim.get('route', '?')} | {lim.get('severity', '?')} | {lim.get('description', '')} |")
    lines.append("")

    FINAL_COMPARISON.write_text("\n".join(lines) + "\n")
    _log(f"Final comparison MD -> {FINAL_COMPARISON}")


def _write_comparison_csv(comparison):
    rows = [["Operator", "Status", "Torch_Correctness", "AscendC_Correctness", "PyPTO_Correctness",
             "Torch_B1_us", "AscendC_B1_us", "PyPTO_B1_us"]]
    for op_name, op_data in comparison.get("operators", {}).items():
        routes = op_data.get("routes", {})
        rows.append([
            op_name,
            op_data.get("final_status", "UNKNOWN"),
            routes.get("torch", {}).get("correctness", "N/A"),
            routes.get("ascendc", {}).get("correctness", "N/A"),
            routes.get("pypto", {}).get("correctness", "N/A"),
            str(routes.get("torch", {}).get("b1_us", "")),
            str(routes.get("ascendc", {}).get("b1_us", "")),
            str(routes.get("pypto", {}).get("b1_us", "")),
        ])

    csv_path = FINAL_DIR / "final_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return csv_path


def _generate_performance_csv():
    release = _load_current_release()
    if not release:
        _log("current_release.json not found, skipping performance CSV")
        return

    rows = [["Operator", "Torch_B1_us", "Torch_B32_us", "AscendC_B1_us", "AscendC_B32_us",
             "PyPTO_B1_us", "PyPTO_B32_us", "Profiler_Method", "Metric"]]
    for op_name, op_data in release.get("operators", {}).items():
        routes = op_data.get("routes", {})
        t = routes.get("torch", {}).get("profiler", {})
        a = routes.get("ascendc", {}).get("profiler", {})
        p = routes.get("pypto", {}).get("profiler", {})
        rows.append([
            op_name,
            t.get("b1_primary_us", ""),
            t.get("b32_primary_us", ""),
            a.get("b1_primary_us", ""),
            a.get("b32_primary_us", ""),
            p.get("b1_primary_us", ""),
            p.get("b32_primary_us", ""),
            t.get("method", "N/A"),
            "primary_compute_kernel_us",
        ])

    PERFORMANCE_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    with open(PERFORMANCE_MATRIX, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    _log(f"Performance CSV -> {PERFORMANCE_MATRIX}")


def _generate_current_release_md():
    release = _load_current_release()
    if not release:
        _log("current_release.json not found, skipping current_release.md")
        return

    lines = [
        f"# Cannbot {release.get('release_version', 'N/A')} Current Release",
        "",
        "**Single source of truth**: `reports/release/current_release.json`",
        f"**Generated**: {release.get('generated_at', 'N/A')}",
        "",
        "## Operator Status",
        "",
        "| Operator | Final Status | Correctness | Profiler |",
        "|----------|-------------|-------------|----------|",
    ]
    for op_name, op_data in release.get("operators", {}).items():
        lines.append(f"| {op_name} | **{op_data.get('final_status', 'UNKNOWN')}** | {op_data.get('correctness_coverage', 'N/A')} | {op_data.get('profiler_coverage', 'N/A')} |")

    lines.append("")
    ss = release.get("status_summary", {})
    lines.append(f"**Status counts**: {', '.join(f'{k}: {v}' for k, v in ss.items())}")
    lines.append("")

    if release.get("known_limitations"):
        lines.append("## Known Limitations")
        lines.append("")
        for lim in release["known_limitations"]:
            lines.append(f"- **{lim.get('operator', '?')}/{lim.get('route', '?')}** [{lim.get('severity', '?')}]: {lim.get('description', '')}")

    CURRENT_RELEASE_MD.write_text("\n".join(lines) + "\n")
    _log(f"Current release MD -> {CURRENT_RELEASE_MD}")


def _log(msg):
    print(f"  [comparison] {msg}")
