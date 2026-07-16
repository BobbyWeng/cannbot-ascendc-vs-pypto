#!/usr/bin/env python3
"""Report generator template for {{ operator_name }}."""
import os, json
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ_DIR, "common"))
from common.reporting import generate_markdown_report, generate_json_report, generate_csv_report

def generate():
    # Load parsed data and build report
    # (Template - fill in with actual data loading logic)
    data = {
        "experiment": {
            "operator": "{{ operator_name }}",
            "shape": "{{ shape }}",
            "dtype": "{{ dtype }}",
        },
        "summary": {
            "status": "PENDING",
            "note": "Report generation not yet implemented for {{ operator_name }}"
        }
    }
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "final")
    generate_json_report(data, os.path.join(report_dir, "final_comparison.json"))
    generate_csv_report(data, os.path.join(report_dir, "final_comparison.csv"))
    generate_markdown_report(data, os.path.join(report_dir, "final_comparison.md"))

if __name__ == "__main__":
    generate()
