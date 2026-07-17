#!/usr/bin/env python3
"""Cannbot Release Pipeline — one-command orchestrator.

Usage:
    python3 scripts/release/release.py              # Full pipeline
    python3 scripts/release/release.py --dry-run    # Dry run (no changes)
    python3 scripts/release/release.py --force      # Force redo all steps
    python3 scripts/release/release.py --step 1     # Run single step
    python3 scripts/release/release.py --skip-sha256  # Skip SHA256 step
"""

import argparse
import json
import sys
import time
from datetime import datetime

from release_config import RELEASE_DIR, RELEASE_MANIFEST, PROJECT_ROOT, RELEASE_VERSION
import step_correctness
import step_profiling
import step_comparison
import step_release as step_release_mod
import step_dashboard
import step_sha256
import step_validate
import step_readme


STEPS = [
    ("correctness", step_correctness, "Run correctness for all operators"),
    ("profiling", step_profiling, "Run msprof + parse for all operators"),
    ("comparison", step_comparison, "Generate final comparison (md/json/csv)"),
    ("release", step_release_mod, "Generate current_release.json + limitation_matrix"),
    ("dashboard", step_dashboard, "Generate dashboard.json from release data"),
    ("sha256", step_sha256, "Regenerate SHA256SUMS for all operators"),
    ("validate", step_validate, "Run regression tests and consistency checks"),
    ("readme", step_readme, "Update operator READMEs"),
]

STEP_NAMES = [s[0] for s in STEPS]


def main():
    parser = argparse.ArgumentParser(description="Cannbot Release Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done, no changes")
    parser.add_argument("--force", action="store_true", help="Force redo all steps")
    parser.add_argument("--step", type=int, choices=range(1, len(STEPS) + 1),
                        help="Run a single step by number")
    parser.add_argument("--skip-sha256", action="store_true", help="Skip SHA256 step")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation step")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip profiling step")
    args = parser.parse_args()

    if args.dry_run:
        _print_header("DRY RUN")

    _print_header(f"Release Pipeline v{RELEASE_VERSION}")

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {}
    all_passed = True

    step_list = STEPS
    if args.step is not None:
        step_list = [STEPS[args.step - 1]]

    if args.skip_sha256:
        step_list = [(n, m, d) for n, m, d in step_list if n != "sha256"]
    if args.skip_validate:
        step_list = [(n, m, d) for n, m, d in step_list if n != "validate"]
    if args.skip_profiling:
        step_list = [(n, m, d) for n, m, d in step_list if n != "profiling"]

    for idx, (name, module, desc) in enumerate(step_list, 1):
        ts = time.strftime("%H:%M:%S")
        print(f"\n[{ts}] Step {idx}/{len(step_list)}: {name} — {desc}")
        print(f"{'─' * 60}")

        try:
            passed = module.run(dry_run=args.dry_run, force=args.force)
        except Exception as e:
            print(f"  [ERROR] {name} raised exception: {e}")
            passed = False

        status = "PASS" if passed else "FAIL"
        results[name] = {"status": status, "timestamp": timestamp}
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print(f"\n{'=' * 60}")
    status_line = "ALL STEPS PASSED" if all_passed else "SOME STEPS FAILED"
    print(f"  {status_line}")
    print(f"{'=' * 60}")

    _write_manifest(results, timestamp, args.dry_run)

    return 0 if all_passed else 1


def _print_header(text):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def _write_manifest(results, timestamp, dry_run):
    manifest = {
        "manifest_version": "1.0",
        "release_version": RELEASE_VERSION,
        "generated_at": timestamp,
        "project_root": str(PROJECT_ROOT),
        "steps": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for v in results.values() if v["status"] == "PASS"),
            "failed": sum(1 for v in results.values() if v["status"] == "FAIL"),
        },
    }

    if not dry_run:
        RELEASE_DIR.mkdir(parents=True, exist_ok=True)
        RELEASE_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"\n  Release manifest -> {RELEASE_MANIFEST}")
    else:
        print(f"\n  [DRY-RUN] Would write release manifest to {RELEASE_MANIFEST}")


if __name__ == "__main__":
    sys.exit(main())
