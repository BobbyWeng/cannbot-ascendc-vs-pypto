#!/usr/bin/env python3
"""Generate all final comparison reports, batch summary, and dashboard."""
import json, os, csv, glob

PROJECT = "/mnt/workspace/cannbot_ascendc_vs_pypto"
BATCHES = [1, 2, 4, 8, 16, 32, 64]

OPERATORS = {
    "equal": {
        "impls": {"torch": "PASS", "ascendc": "PASS", "pypto": "BLOCKED_BACKEND_EQUAL"},
        "status": "COMPLETE_WITH_LIMITATION",
    },
    "where": {
        "impls": {"torch": "PASS", "ascendc": "PASS", "pypto": "BLOCKED_BACKEND_WHERE_SELECT"},
        "status": "COMPLETE_WITH_LIMITATION",
    },
    "not": {
        "impls": {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
        "status": "COMPLETE",
    },
    "or": {
        "impls": {"torch": "PASS", "ascendc": "PASS", "pypto": "PASS"},
        "status": "COMPLETE",
    },
}


def load_result(op, impl):
    """Load profiler result for an operator/implementation. Returns dict per batch."""
    results = {}
    for b in BATCHES:
        path = f"{PROJECT}/operators/{op}/reports/raw/{impl}/b{b}/result.json"
        if os.path.exists(path):
            with open(path) as f:
                results[b] = json.load(f)
    return results


def generate_final_report(op):
    op_info = OPERATORS[op]
    all_results = {}
    for impl in op_info["impls"]:
        all_results[impl] = load_result(op, impl)

    lines = []
    lines.append(f"# {op.title()} — Final Comparison Report")
    lines.append("")
    lines.append("## Spec")
    lines.append(f"- Operator: {op}")
    lines.append("- Shape: [B, 256, 384]")
    if op == "equal":
        lines.append("- Input dtype: float16, Output dtype: uint8 (BOOL)")
        lines.append("- Computation: torch.eq(x1, x2)")
    elif op == "where":
        lines.append("- Input: condition uint8, x1/x2 float16, Output: float16")
        lines.append("- Computation: torch.where(condition, x1, x2)")
    elif op == "not":
        lines.append("- Input/Output dtype: uint8")
        lines.append("- Computation: torch.logical_not(x)")
    elif op == "or":
        lines.append("- Input/Output dtype: uint8 (0/1 normalized)")
        lines.append("- Computation: torch.logical_or(x1, x2)")
    lines.append("")

    lines.append("## Correctness")
    for impl, status in op_info["impls"].items():
        if status.startswith("BLOCKED"):
            lines.append(f"- **{impl}**: {status}")
        else:
            lines.append(f"- **{impl}**: PASS")
    lines.append("")

    lines.append("## Implementation Status")
    lines.append(f"- Overall: {op_info['status']}")
    for impl, status in op_info["impls"].items():
        pt = " (no profiler)" if status.startswith("BLOCKED") else ""
        lines.append(f"- **{impl}**: {status}{pt}")
    lines.append("")

    lines.append("## Profiler Configuration")
    lines.append("- Warmup: 200 iterations")
    lines.append("- Profiled loops: 100")
    lines.append("- Repeat: 5")
    lines.append("- Metric: host-synchronized operation (torch.npu.Event)")
    lines.append("- Method: two-process (warmup → timed loop)")
    lines.append("")

    lines.append("## Results (primary_compute_kernel_us equivalent)")
    lines.append("")
    lines.append("| Batch | " + " | ".join(op_info["impls"].keys()) + " |")
    lines.append("|-------|" + "|".join("---" for _ in op_info["impls"]) + "|")
    for b in BATCHES:
        row = f"| B={b} "
        for impl in op_info["impls"]:
            status = op_info["impls"][impl]
            if status.startswith("BLOCKED"):
                row += " | N/A "
            elif b in all_results.get(impl, {}):
                r = all_results[impl][b]
                row += f" | {r.get('median_us', '?'):.1f} us "
            else:
                row += " | ? "
        row += "|"
        lines.append(row)
    lines.append("")

    lines.append("## Kernel Details")
    for impl in op_info["impls"]:
        status = op_info["impls"][impl]
        if status.startswith("BLOCKED"):
            lines.append(f"\n### {impl}")
            lines.append(f"- {status}")
            continue
        lines.append(f"\n### {impl}")
        for b in BATCHES:
            if b not in all_results.get(impl, {}):
                continue
            r = all_results[impl][b]
            kn = r.get("kernel_info", {}).get("kernel_names", ["unknown"])
            lines.append(f"- B={b}: kernel(s)={kn}, median={r.get('median_us', '?'):.1f} us, "
                         f"mean={r.get('mean_us', '?'):.1f} us, min={r.get('min_us', '?'):.1f} us, "
                         f"P90={r.get('p90_us', '?'):.1f} us, std={r.get('std_us', '?'):.2f}, "
                         f"CV={r.get('cv', '?'):.1f}%")
    lines.append("")

    lines.append("## Known Limitations")
    if op in ("equal", "where"):
        lines.append(f"- **PyPTO {op}**: {op_info['impls']['pypto']} — blocked at backend, not in performance ranking")
    if op == "or":
        lines.append("- PyPTO uses bitwise_or on 0/1 normalized uint8. For non-0/1 inputs, results differ from logical_or.")
    lines.append("")

    lines.append("## Reproduction")
    lines.append("```bash")
    for impl in op_info["impls"]:
        if op_info["impls"][impl].startswith("BLOCKED"):
            continue
        if impl == "torch":
            lines.append(f"# Torch: python3 operators/{op}/torch/benchmark.py")
        elif impl == "ascendc":
            lines.append(f"# Ascend C: operators/{op}/ascendc/build/{op}_ascendc 0 <B> 20 8192 200 100 5")
        elif impl == "pypto":
            lines.append(f"# PyPTO: python3 operators/{op}/pypto/tests/test_{op}.py")
    lines.append("```")
    lines.append("")

    report_text = "\n".join(lines)
    return report_text


def generate_comparison_json(op):
    op_info = OPERATORS[op]
    all_results = {}
    for impl in op_info["impls"]:
        all_results[impl] = load_result(op, impl)

    report = {
        "operator": op,
        "overall_status": op_info["status"],
        "correctness": {},
        "profiler": {},
    }
    for impl, status in op_info["impls"].items():
        report["correctness"][impl] = status
        if not status.startswith("BLOCKED"):
            report["profiler"][impl] = {
                str(b): {
                    "median_us": all_results.get(impl, {}).get(b, {}).get("median_us", None),
                    "mean_us": all_results.get(impl, {}).get(b, {}).get("mean_us", None),
                    "min_us": all_results.get(impl, {}).get(b, {}).get("min_us", None),
                    "p90_us": all_results.get(impl, {}).get(b, {}).get("p90_us", None),
                    "cv": all_results.get(impl, {}).get(b, {}).get("cv", None),
                    "raw_repeat_latency_us": all_results.get(impl, {}).get(b, {}).get("raw_repeat_latency_us", []),
                }
                for b in BATCHES if b in all_results.get(impl, {})
            }
        else:
            report["profiler"][impl] = {"blocked": True, "reason": status}
    return report


def generate_comparison_csv(op):
    op_info = OPERATORS[op]
    all_results = {}
    for impl in op_info["impls"]:
        all_results[impl] = load_result(op, impl)

    rows = []
    header = ["Batch"]
    for impl in op_info["impls"]:
        header.append(f"{impl}_median_us")
        header.append(f"{impl}_mean_us")
        header.append(f"{impl}_cv")
    rows.append(header)

    for b in BATCHES:
        row = [str(b)]
        for impl in op_info["impls"]:
            if op_info["impls"][impl].startswith("BLOCKED"):
                row.extend(["N/A", "N/A", "N/A"])
            elif b in all_results.get(impl, {}):
                r = all_results[impl][b]
                row.extend([
                    f"{r.get('median_us', '?'):.1f}",
                    f"{r.get('mean_us', '?'):.1f}",
                    f"{r.get('cv', '?'):.1f}",
                ])
            else:
                row.extend(["?", "?", "?"])
        rows.append(row)
    return rows


def update_batch_summary():
    """Update reports/batches/logical_ops_v1/ files."""
    batch_dir = f"{PROJECT}/reports/batches/logical_ops_v1"

    # profiler_matrix.csv
    rows = [["Operator", "Implementation", "B1_us", "B2_us", "B4_us", "B8_us", "B16_us", "B32_us", "B64_us"]]
    for op in ["equal", "where", "not", "or"]:
        for impl in OPERATORS[op]["impls"]:
            if OPERATORS[op]["impls"][impl].startswith("BLOCKED"):
                rows.append([op, impl, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
            else:
                r = load_result(op, impl)
                row = [op, impl]
                for b in BATCHES:
                    row.append(f"{r.get(b, {}).get('median_us', '?'):.1f}")
                rows.append(row)
    with open(f"{batch_dir}/profiler_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)

    # benchmark_summary.json
    summary = {"batch": "logical_ops_v1", "generated_at": "2026-07-16T16:30:00Z", "operators": {}}
    for op in ["equal", "where", "not", "or"]:
        summary["operators"][op] = {"status": OPERATORS[op]["status"], "implementations": {}}
        for impl in OPERATORS[op]["impls"]:
            status = OPERATORS[op]["impls"][impl]
            if status.startswith("BLOCKED"):
                summary["operators"][op]["implementations"][impl] = {"status": status, "profiler": "N/A"}
            else:
                r = load_result(op, impl)
                impl_data = {"status": status}
                for b in BATCHES:
                    if b in r:
                        impl_data[str(b)] = {"median_us": r[b].get("median_us")}
                summary["operators"][op]["implementations"][impl] = impl_data

    with open(f"{batch_dir}/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # benchmark_summary.md
    md_lines = ["# logical_ops_v1 Benchmark Summary", "",
                "Generated: 2026-07-16T16:30:00Z", "",
                "## Profiler Matrix", ""]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    if len(rows) > 1:
        md_lines.insert(len(md_lines) - len(rows) + 1,
                       "|" + "|".join("---" for _ in rows[0]) + "|")
    with open(f"{batch_dir}/benchmark_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    # Update final_report.json
    final_path = f"{batch_dir}/final_report.json"
    with open(final_path) as f:
        fr = json.load(f)
    for op in ["equal", "where", "not", "or"]:
        if "profiler" not in fr["summary"][op]:
            fr["summary"][op]["profiler"] = {}
        for impl in OPERATORS[op]["impls"]:
            status = OPERATORS[op]["impls"][impl]
            r = load_result(op, impl)
            fr["summary"][op]["profiler"][impl] = {
                "status": "COMPLETE" if not status.startswith("BLOCKED") else "BLOCKED"
            }
            if not status.startswith("BLOCKED") and r:
                fr["summary"][op]["profiler"][impl]["batches"] = len(r)
                fr["summary"][op]["profiler"][impl]["b1_median_us"] = r.get(1, {}).get("median_us")
    with open(final_path, "w") as f:
        json.dump(fr, f, indent=2)

    # Final operator matrix
    op_rows = [["Operator", "Torch", "Ascend C", "PyPTO", "Overall Status"]]
    for op in ["equal", "where", "not", "or"]:
        op_rows.append([
            op,
            f"PASS + profiler ({load_result(op, 'torch').get(1, {}).get('median_us', '?'):.1f} us @ B=1)",
            f"PASS + profiler ({load_result(op, 'ascendc').get(1, {}).get('median_us', '?'):.1f} us @ B=1)",
            OPERATORS[op]["impls"]["pypto"],
            OPERATORS[op]["status"]
        ])
    with open(f"{batch_dir}/final_operator_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(op_rows)

    print("Batch summary files updated")


def update_dashboard():
    """Update dashboard/dashboard.json and index.html."""
    dashboard = {"operators": {}, "batches": {"logical_ops_v1": {}}}
    for op in ["equal", "where", "not", "or"]:
        op_entry = {
            "status": OPERATORS[op]["status"],
            "correctness": {},
            "profiler": {},
        }
        for impl, status in OPERATORS[op]["impls"].items():
            op_entry["correctness"][impl] = "PASS" if not status.startswith("BLOCKED") else status
            r = load_result(op, impl)
            if not status.startswith("BLOCKED") and r:
                op_entry["profiler"][impl] = {"complete": True, "b1_us": r.get(1, {}).get("median_us")}
            else:
                op_entry["profiler"][impl] = {"complete": False, "blocked": True}
        dashboard["operators"][op] = op_entry
        dashboard["batches"]["logical_ops_v1"][op] = op_entry["status"]

    os.makedirs(f"{PROJECT}/dashboard", exist_ok=True)
    with open(f"{PROJECT}/dashboard/dashboard.json", "w") as f:
        json.dump(dashboard, f, indent=2)

    # Generate index.html
    html = """<!DOCTYPE html><html><head><title>Cannbot Dashboard</title>
<style>body{font-family:monospace;margin:20px}table{border-collapse:collapse}
th,td{border:1px solid #ccc;padding:8px;text-align:center}
.pass{background:#dfd}.fail{background:#fdd}.blocked{background:#ffd}
.limited{background:#ffe0b0}</style></head><body>
<h1>Cannbot: Ascend C vs PyPTO — Dashboard</h1>
"""
    html += "<h2>logical_ops_v1</h2>\n<table>\n<tr><th>Operator</th><th>Overall</th>"
    for impl in ["torch", "ascendc", "pypto"]:
        html += f"<th>{impl}</th>"
    html += "<th>B1 median (us)</th></tr>\n"

    for op in ["equal", "where", "not", "or"]:
        status = OPERATORS[op]["status"]
        cls = {"COMPLETE": "pass", "COMPLETE_WITH_LIMITATION": "limited",
               "INCOMPLETE": "fail", "BLOCKED": "blocked"}.get(status, "")
        html += f'<tr><td>{op}</td><td class="{cls}">{status}</td>'
        for impl in ["torch", "ascendc", "pypto"]:
            st = OPERATORS[op]["impls"][impl]
            cls2 = "pass" if st == "PASS" else ("blocked" if "BLOCKED" in st else "fail")
            html += f'<td class="{cls2}">{st}</td>'
        # B1 median
        r_t = load_result(op, "torch").get(1, {})
        r_a = load_result(op, "ascendc").get(1, {})
        us_t = r_t.get("median_us", "N/A")
        us_a = r_a.get("median_us", "N/A")
        if us_t != "N/A" and us_a != "N/A":
            html += f"<td>Torch: {us_t:.1f}<br>AscendC: {us_a:.1f}</td>"
        else:
            html += "<td>N/A</td>"
        html += "</tr>\n"

    html += "</table>\n<h2>Archives</h2>\n<ul>\n"
    for op in ["equal", "where", "not", "or"]:
        html += f"<li>cannbot_ascendc_vs_pypto_{op}_v4.tar.gz</li>\n"
    html += "</ul>\n</body></html>"

    with open(f"{PROJECT}/dashboard/index.html", "w") as f:
        f.write(html)
    print("Dashboard updated")


def update_readmes():
    """Update README.md files."""
    for op in ["equal", "where", "not", "or"]:
        path = f"{PROJECT}/operators/{op}/README.md"
        r_t = load_result(op, "torch").get(1, {})
        r_a = load_result(op, "ascendc").get(1, {})
        us_t = f"{r_t.get('median_us', '?'):.1f}" if r_t else "N/A"
        us_a = f"{r_a.get('median_us', '?'):.1f}" if r_a else "N/A"
        content = f"""# {op.title()} Operator

## Implementation Status: {OPERATORS[op]['status']}

| Implementation | Correctness | B1 Latency |
|----------------|-------------|------------|"""
        for impl in OPERATORS[op]["impls"]:
            st = OPERATORS[op]["impls"][impl]
            if st.startswith("BLOCKED"):
                content += f"\n| {impl} | {st} | N/A |"
            else:
                us = "?"
                r = load_result(op, impl).get(1, {})
                if r:
                    us = f"{r.get('median_us', '?'):.1f} us"
                content += f"\n| {impl} | PASS | {us} |"

        content += "\n## Kernel Details\n"
        for impl in OPERATORS[op]["impls"]:
            if OPERATORS[op]["impls"][impl].startswith("BLOCKED"):
                content += f"\n{impl}: {OPERATORS[op]['impls'][impl]}\n"
            else:
                r = load_result(op, impl)
                b1 = r.get(1, {})
                content += f"\n{impl}: kernel={b1.get('kernel_info', {}).get('kernel_names', ['?'])}, B1={b1.get('median_us', '?'):.1f} us\n"
        with open(path, "w") as f:
            f.write(content)

    # Operators README
    op_readme = "# Operator Summary\n\n"
    for op in ["equal", "where", "not", "or"]:
        st = OPERATORS[op]["status"]
        r_t = load_result(op, "torch").get(1, {})
        r_a = load_result(op, "ascendc").get(1, {})
        op_readme += f"- **{op}**: {st} (Torch: {r_t.get('median_us', '?'):.1f} us, AscendC: {r_a.get('median_us', '?'):.1f} us, PyPTO: {OPERATORS[op]['impls']['pypto']})\n"
    with open(f"{PROJECT}/operators/README.md", "w") as f:
        f.write(op_readme)

    # Report summary
    rpt = "# Operator Summary\n\n"
    for op in ["equal", "where", "not", "or"]:
        rpt += f"## {op}\n- Status: {OPERATORS[op]['status']}\n"
        for impl in OPERATORS[op]["impls"]:
            rpt += f"  - {impl}: {OPERATORS[op]['impls'][impl]}\n"
        rpt += "\n"
    with open(f"{PROJECT}/reports/operator_summary.md", "w") as f:
        f.write(rpt)

    rpt_json = {"generated_at": "2026-07-16T16:30:00Z"}
    for op in ["equal", "where", "not", "or"]:
        rpt_json[op] = {"status": OPERATORS[op]["status"]}
        for impl in OPERATORS[op]["impls"]:
            rpt_json[op][impl] = OPERATORS[op]["impls"][impl]
    with open(f"{PROJECT}/reports/operator_summary.json", "w") as f:
        json.dump(rpt_json, f, indent=2)

    print("README files updated")


if __name__ == "__main__":
    for op in ["equal", "where", "not", "or"]:
        report_dir = f"{PROJECT}/operators/{op}/reports/final"
        os.makedirs(report_dir, exist_ok=True)

        md = generate_final_report(op)
        with open(f"{report_dir}/final_comparison.md", "w") as f:
            f.write(md)
        print(f"Generated {op}/final_comparison.md")

        j = generate_comparison_json(op)
        with open(f"{report_dir}/final_comparison.json", "w") as f:
            json.dump(j, f, indent=2)

        csv_rows = generate_comparison_csv(op)
        with open(f"{report_dir}/final_comparison.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(csv_rows)
        print(f"Generated {op}/final_comparison.csv")

    update_batch_summary()
    update_dashboard()
    update_readmes()
    print("\nAll reports generated successfully!")
