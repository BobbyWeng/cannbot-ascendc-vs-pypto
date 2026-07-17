import json
import sys
import subprocess
from release_config import (
    OPERATORS, OPERATOR_DIR, REGRESSION_DIR, RELEASE_DIR,
    DASHBOARD_JSON,
)

def run(dry_run=False, force=False):
    if dry_run:
        print("[DRY-RUN] step_validate: would run regression tests")
        return True

    errors = []

    _log("running regression test suite...")
    result = subprocess.run(
        ["bash", str(REGRESSION_DIR / "run_regression.sh"), "--json"],
        capture_output=True, text=True, cwd=REGRESSION_DIR.parent.parent,
    )
    _log(f"regression exit code: {result.returncode}")
    for line in result.stdout.splitlines():
        _log(f"  {line}")

    release_data = None
    release_json = RELEASE_DIR / "current_release.json"
    if release_json.exists():
        release_data = json.loads(release_json.read_text())

    if release_data:
        op_count = release_data.get("operator_count", 0)
        if op_count != len(OPERATORS):
            errors.append(f"operator_count={op_count}, expected {len(OPERATORS)}")

        ops = release_data.get("operators", {})
        for op in OPERATORS:
            if op not in ops:
                errors.append(f"missing operator: {op}")

    if DASHBOARD_JSON.exists():
        try:
            dash = json.loads(DASHBOARD_JSON.read_text())
            dash_ops = dash.get("operators", {})
            for op in OPERATORS:
                if op not in dash_ops:
                    errors.append(f"dashboard missing operator: {op}")
        except Exception as e:
            errors.append(f"dashboard JSON parse failed: {e}")

    status = "PASS" if not errors else "FAIL"
    _log(f"validation {status} ({len(errors)} error(s))")
    for e in errors:
        _log(f"  ERROR: {e}")

    return len(errors) == 0


def _log(msg):
    print(f"  [validate] {msg}")
