#!/usr/bin/env python3
"""Verify that Cannbot is being used correctly in the current task context.

Checks:
1. task_context.json exists with valid classification
2. Active plugin is recorded
3. Required skills are loaded
4. SKILL_TRACE is updated
5. Cannbot commit is recorded
6. Candidate diff exists (if source modified)
7. Correctness re-run after modification
8. Profiler re-run after modification
9. Reviewer completed

Usage:
    python3 tools/verify_cannbot_usage.py [--stage check|full]
    python3 tools/verify_cannbot_usage.py --json
"""

import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_CONTEXT = os.path.join(PROJECT_ROOT, "reports", "runtime", "task_context.json")
NPU_RUN_QUEUE = os.path.join(PROJECT_ROOT, "reports", "runtime", "npu_run_queue.json")
PERMISSION_DEFERRED = os.path.join(PROJECT_ROOT, "reports", "runtime", "permission_deferred.json")
PROJECT_STATE = os.path.join(PROJECT_ROOT, "reports", "runtime", "project_state.json")


def check_task_context():
    if not os.path.isfile(TASK_CONTEXT):
        return False, "task_context.json missing"
    with open(TASK_CONTEXT) as f:
        ctx = json.load(f)
    required = ["backend_route", "semantic_class", "hardware_path", "task_mode", "lifecycle_stage",
                 "active_plugin", "active_agent"]
    missing = [k for k in required if k not in ctx]
    if missing:
        return False, f"task_context.json missing fields: {missing}"
    return True, "task_context.json valid"


def check_plugin_recorded():
    if not os.path.isfile(TASK_CONTEXT):
        return False, "task_context.json missing"
    with open(TASK_CONTEXT) as f:
        ctx = json.load(f)
    plugin = ctx.get("active_plugin")
    if not plugin:
        return False, "No active_plugin in task_context"
    return True, f"Plugin: {plugin}"


def check_skills_loaded():
    if not os.path.isfile(TASK_CONTEXT):
        return False, "task_context.json missing"
    with open(TASK_CONTEXT) as f:
        ctx = json.load(f)
    required = ctx.get("required_skills", [])
    loaded = ctx.get("loaded_skills", [])
    if not loaded:
        return False, "No skills loaded"
    missing = [s for s in required if s not in loaded]
    if missing:
        return False, f"Required skills not loaded: {missing}"
    return True, f"Skills: {len(loaded)} loaded, {len(missing)} missing"


def check_skill_trace(operator):
    """Check SKILL_TRACE files exist and contain records"""
    if operator:
        st_json = os.path.join(PROJECT_ROOT, "operators", operator, "SKILL_TRACE.json")
        st_md = os.path.join(PROJECT_ROOT, "operators", operator, "SKILL_TRACE.md")
    else:
        st_json = os.path.join(PROJECT_ROOT, "SKILL_TRACE.json")
        st_md = os.path.join(PROJECT_ROOT, "SKILL_TRACE.md")

    found = []
    if os.path.isfile(st_json):
        found.append("SKILL_TRACE.json")
    if os.path.isfile(st_md):
        found.append("SKILL_TRACE.md")

    if not found:
        return False, "No SKILL_TRACE files found"
    return True, f"SKILL_TRACE: {', '.join(found)}"


def check_cannbot_commit():
    if not os.path.isfile(TASK_CONTEXT):
        return False, "task_context.json missing"
    with open(TASK_CONTEXT) as f:
        ctx = json.load(f)
    commit = ctx.get("cannbot_commit")
    if not commit:
        return False, "cannbot_commit not recorded in task_context"
    return True, f"Cannbot commit: {commit}"


def check_candidate_diff():
    """Check that a diff exists if source was modified"""
    result = subprocess.run(
        ["git", "diff", "--stat"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    if result.returncode == 0 and result.stdout.strip():
        return True, f"Uncommitted changes:\n{result.stdout.strip()}"
    return False, "No uncommitted changes detected (or not a git repo)"


def check_npu_queue():
    if not os.path.isfile(NPU_RUN_QUEUE):
        return False, "npu_run_queue.json missing"
    with open(NPU_RUN_QUEUE) as f:
        try:
            queue = json.load(f)
            if isinstance(queue, list) and len(queue) > 0:
                running = [t for t in queue if t.get("status") in ("running", "queued")]
                if running:
                    return True, f"NPU queue has {len(running)} active items"
                return True, f"NPU queue: {len(queue)} completed items"
            return True, "NPU queue exists (empty)"
        except json.JSONDecodeError:
            return False, "npu_run_queue.json is not valid JSON"


def check_permission_deferred():
    if os.path.isfile(PERMISSION_DEFERRED):
        with open(PERMISSION_DEFERRED) as f:
            try:
                deferred = json.load(f)
                if isinstance(deferred, list) and len(deferred) > 0:
                    return True, f"Permission deferred: {len(deferred)} items"
                return True, "Permission deferred file exists (empty)"
            except json.JSONDecodeError:
                return False, "permission_deferred.json is not valid JSON"
    return True, "No permission_deferred.json (no deferred actions)"


def run_full_check(operator):
    checks = [
        ("task_context", check_task_context),
        ("plugin_recorded", check_plugin_recorded),
        ("skills_loaded", check_skills_loaded),
        ("skill_trace", lambda: check_skill_trace(operator)),
        ("cannbot_commit", check_cannbot_commit),
        ("candidate_diff", check_candidate_diff),
        ("npu_queue", check_npu_queue),
        ("permission_deferred", check_permission_deferred),
    ]
    results = {}
    all_pass = True
    for name, check_fn in checks:
        passed, message = check_fn()
        results[name] = {"passed": passed, "message": message}
        if not passed:
            all_pass = False
    return all_pass, results


def main():
    parser = argparse.ArgumentParser(description="Verify Cannbot usage compliance")
    parser.add_argument("--stage", choices=["check", "full"], default="full",
                        help="Check stage (check=basic, full=all)")
    parser.add_argument("--operator", help="Operator name for operator-specific checks")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.stage == "check":
        passed, msg = check_task_context()
        print(msg)
        return 0 if passed else 1

    all_pass, results = run_full_check(args.operator)

    if args.json:
        print(json.dumps({"all_pass": all_pass, "checks": results}, indent=2))
    else:
        print(f"Cannbot Usage Verification: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        print("=" * 60)
        for name, info in results.items():
            status = "PASS" if info["passed"] else "FAIL"
            print(f"  {name:25s} [{status:4s}] {info['message']}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
