import json
import subprocess
import sys
from release_config import OPERATORS, OPERATOR_DIR, SCRIPTS_DIR

def run(dry_run=False, force=False):
    if dry_run:
        print("[DRY-RUN] step_profiling: would run msprof + parse for all operators")
        return True

    for op in OPERATORS:
        _run_operator_profiling(op, force)
    return True


def _run_operator_profiling(op, force):
    op_dir = OPERATOR_DIR / op
    parsed_dir = op_dir / "reports" / "parsed"
    raw_dir = op_dir / "reports" / "raw"

    if not force and parsed_dir.exists() and list(parsed_dir.glob("*.json")):
        _log(f"{op}: parsed data exists, skipping (use --force to redo)")
        return True

    run_all = op_dir / "benchmark" / "run_all.sh"
    if run_all.exists():
        _log(f"{op}: running benchmark/run_all.sh ...")
        result = subprocess.run(["bash", str(run_all)], capture_output=True, text=True, cwd=op_dir)
        if result.returncode != 0:
            _log(f"{op}: benchmark failed (rc={result.returncode}), stderr follows:")
            for line in result.stderr.splitlines():
                _log(f"  {line}")
            return False
        _log(f"{op}: benchmark completed")
        return True

    logical_script = SCRIPTS_DIR / "run_msprof_logical_ops.sh"
    if logical_script.exists():
        _log(f"{op}: found in logical ops script, will run via queue")
        return True

    _log(f"{op}: no profiling script found, skipping")
    return True


def _log(msg):
    print(f"  [profiling] {msg}")
