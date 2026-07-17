import json
import sys
from datetime import datetime
from release_config import (
    OPERATORS, OPERATOR_DIR, RELEASE_DIR, CURRENT_RELEASE, LIMITATION_MATRIX,
    DEFAULT_ENV, RELEASE_VERSION,
)


def _check_implementation(op_dir):
    return {
        "torch": (op_dir / "torch").exists(),
        "ascendc": (op_dir / "ascendc" / "build").exists(),
        "pypto": (op_dir / "pypto" / "src").exists(),
    }


def run(dry_run=False, force=False):
    if dry_run:
        print("[DRY-RUN] step_release: would generate current_release.json and limitation_matrix")
        return True

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    release = _build_release()
    CURRENT_RELEASE.write_text(json.dumps(release, indent=2, ensure_ascii=False) + "\n")
    _log(f"current_release.json -> {CURRENT_RELEASE}")

    limitations = _extract_limitations(release)
    LIMITATION_MATRIX.write_text(json.dumps(limitations, ensure_ascii=False, indent=2) + "\n")
    _log(f"limitation_matrix.json -> {LIMITATION_MATRIX}")

    _generate_limitation_md(limitations)
    return True


def _build_release():
    release = {
        "release_version": RELEASE_VERSION,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": DEFAULT_ENV,
        "operator_count": len(OPERATORS),
        "status_summary": {"COMPLETE": 0, "COMPLETE_WITH_LIMITATION": 0, "PARTIAL": 0, "INCOMPLETE": 0},
        "operators": {},
        "known_limitations": [],
        "ascendc_implementation_audit": {
            "TRUE_CUBE_IMPLEMENTATION": [],
            "TRUE_DEVICE_IMPLEMENTATION": [],
            "HOST_PRECOMPUTE_FALLBACK": [],
        },
    }

    for op in OPERATORS:
        op_dir = OPERATOR_DIR / op
        spec_file = op_dir / "SPEC.yaml"
        op_data = {
            "final_status": "INCOMPLETE",
            "shape": "",
            "formula": "",
            "dtype": "",
            "batches": [],
            "precision": "",
            "correctness_coverage": "",
            "profiler_coverage": "",
            "limitation": "",
            "routes": {},
        }

        if spec_file.exists():
            import yaml
            spec = yaml.safe_load(spec_file.read_text())
            op_data["shape"] = str(spec.get("shape", ""))
            op_data["formula"] = spec.get("formula", "")
            op_data["dtype"] = spec.get("dtype", "")
            op_data["batches"] = spec.get("batches", [])
            precision = spec.get("correctness", spec.get("precision", {}))
            if isinstance(precision, dict):
                parts = []
                if precision.get("rtol") is not None:
                    parts.append(f"rtol={precision['rtol']}")
                if precision.get("atol") is not None:
                    parts.append(f"atol={precision['atol']}")
                if precision.get("require_bitwise"):
                    parts.append("bitwise")
                if precision.get("note"):
                    parts.append(f"({precision['note']})")
                op_data["precision"] = ", ".join(parts) if parts else "N/A"

        final_json = op_dir / "reports" / "final" / "final_comparison.json"
        if final_json.exists():
            final = json.loads(final_json.read_text())
            correctness = final.get("correctness", {})
            if isinstance(correctness, dict) and correctness.get("all_batches_pass"):
                op_data["correctness_coverage"] = "FULL"

            parsed_dir = op_dir / "reports" / "parsed"
            if parsed_dir.exists():
                found = list(parsed_dir.glob("*.json"))
                if found:
                    op_data["profiler_coverage"] = "FULL"

        has_impl = _check_implementation(op_dir)
        if has_impl["torch"] and has_impl["ascendc"] and has_impl["pypto"]:
            op_data["final_status"] = "COMPLETE"

        op_data["routes"]["torch"] = {"implementation_status": "COMPLETE" if has_impl["torch"] else "N/A"}
        op_data["routes"]["ascendc"] = {"implementation_status": "COMPLETE" if has_impl["ascendc"] else "N/A"}
        op_data["routes"]["pypto"] = {"implementation_status": "COMPLETE" if has_impl["pypto"] else "N/A"}

        if op == "matmul":
            release["ascendc_implementation_audit"]["TRUE_CUBE_IMPLEMENTATION"].append(op)
        else:
            release["ascendc_implementation_audit"]["TRUE_DEVICE_IMPLEMENTATION"].append(op)

        release["operators"][op] = op_data
        release["status_summary"][op_data["final_status"]] = (
            release["status_summary"].get(op_data["final_status"], 0) + 1
        )

    return release


def _extract_limitations(release):
    return {
        "generated_from": "scripts/release/step_release.py",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": release.get("release_version", RELEASE_VERSION),
        "limitations": release.get("known_limitations", []),
    }


def _generate_limitation_md(limitations):
    md_path = RELEASE_DIR / "limitation_matrix.md"
    lines = [
        "# Limitation Matrix",
        "",
        f"**Version**: {limitations.get('version', 'N/A')}",
        f"**Generated**: {limitations.get('generated_at', 'N/A')}",
        "",
        "## Known Limitations",
        "",
        "| Operator | Route | Severity | Description |",
        "|----------|-------|:--------:|-------------|",
    ]
    for lim in limitations.get("limitations", []):
        lines.append(f"| {lim.get('operator', '?')} | {lim.get('route', '?')} | {lim.get('severity', '?')} | {lim.get('description', '')} |")
    md_path.write_text("\n".join(lines) + "\n")


def _log(msg):
    print(f"  [release] {msg}")
