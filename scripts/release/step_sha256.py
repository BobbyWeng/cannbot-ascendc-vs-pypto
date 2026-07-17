import sys
import subprocess
from release_config import OPERATORS, OPERATOR_DIR, SHA256SUMS_FILE

def run(dry_run=False, force=False):
    if dry_run:
        print("[DRY-RUN] step_sha256: would regenerate SHA256SUMS for all operators")
        return True

    for op in OPERATORS:
        op_dir = OPERATOR_DIR / op
        _regenerate_sha256(op_dir, op, force)

    _log("SHA256SUMS regeneration complete")
    return True


def _regenerate_sha256(op_dir, op, force):
    sha_path = op_dir / SHA256SUMS_FILE
    if not force and _validate_existing(sha_path, op_dir):
        _log(f"{op}: valid SHA256SUMS exists, skipping")
        return True

    _log(f"{op}: regenerating SHA256SUMS...")
    result = subprocess.run(
        ["bash", "-c", "find . -type f ! -path './.git/*' ! -path './reports/raw/*' ! -name 'SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum"],
        capture_output=True, text=True, cwd=op_dir,
    )
    if result.returncode == 0:
        sha_path.write_text(result.stdout)
        _log(f"{op}: SHA256SUMS regenerated ({len(result.stdout.splitlines())} files)")
        return True
    else:
        _log(f"{op}: sha256sum generation failed (rc={result.returncode})")
        return False


def _validate_existing(sha_path, op_dir):
    if not sha_path.exists():
        return False
    result = subprocess.run(
        ["sha256sum", "-c", str(sha_path)],
        capture_output=True, text=True, cwd=op_dir,
    )
    return result.returncode == 0


def _log(msg):
    print(f"  [sha256] {msg}")
