import sys
import subprocess
from release_config import DASHBOARD_DIR, DASHBOARD_JSON, DASHBOARD_HTML, PROJECT_ROOT

def run(dry_run=False, force=False):
    if dry_run:
        print("[DRY-RUN] step_dashboard: would generate dashboard.json from release data")
        return True

    dashboard_py = DASHBOARD_DIR / "dashboard.py"
    if not dashboard_py.exists():
        _log("dashboard.py not found, skipping")
        return True

    release_json = PROJECT_ROOT / "reports" / "release" / "current_release.json"
    if not release_json.exists():
        _log("current_release.json not found, skipping dashboard")
        return True

    cmd = ["python3", str(dashboard_py), "--release", str(release_json)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=DASHBOARD_DIR)
    for line in result.stdout.splitlines():
        _log(line)
    if result.returncode != 0:
        _log(f"dashboard generation failed (rc={result.returncode})")
        for line in result.stderr.splitlines():
            _log(f"  {line}")
        return False

    if DASHBOARD_JSON.exists():
        _log(f"dashboard.json -> {DASHBOARD_JSON}")
    if DASHBOARD_HTML.exists():
        _log(f"index.html -> {DASHBOARD_HTML}")
    return True


def _log(msg):
    print(f"  [dashboard] {msg}")
