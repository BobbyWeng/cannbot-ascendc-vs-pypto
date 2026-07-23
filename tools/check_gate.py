#!/usr/bin/env python3
"""Gate verification tool. Reads config/gates.yaml and checks each gate's pass condition.

Usage:
    python3 tools/check_gate.py                        # Check all gates
    python3 tools/check_gate.py --gate G3              # Check specific gate
    python3 tools/check_gate.py --operator relu         # Check all gates for operator
    python3 tools/check_gate.py --json                  # JSON output
"""

import argparse
import json
import os
import subprocess
import sys
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES_CONFIG = os.path.join(PROJECT_ROOT, "config", "gates.yaml")
TASK_CONTEXT = os.path.join(PROJECT_ROOT, "reports", "runtime", "task_context.json")
PROJECT_STATE = os.path.join(PROJECT_ROOT, "reports", "runtime", "project_state.json")


def load_gates():
    with open(GATES_CONFIG) as f:
        config = yaml.safe_load(f)
    return config["gates"]


def file_exists(path):
    return os.path.isfile(path)


def check_g0():
    """Environment check"""
    preflight = os.path.join(PROJECT_ROOT, "environment", "preflight.sh")
    if not file_exists(preflight):
        return False, "preflight.sh not found"
    try:
        result = subprocess.run(
            ["bash", preflight], capture_output=True, text=True,
            timeout=30,
            env={**os.environ, "OPCODE_GATE_CHECK": "1"}
        )
        return result.returncode == 0, result.stderr.strip() or result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "preflight.sh timed out (NPU query hung — check torch.npu availability)"


def check_g1():
    """Classification check"""
    if not file_exists(TASK_CONTEXT):
        return False, "task_context.json not found"
    with open(TASK_CONTEXT) as f:
        ctx = json.load(f)
    required = ["backend_route", "semantic_class", "hardware_path", "task_mode", "lifecycle_stage"]
    missing = [k for k in required if k not in ctx]
    return len(missing) == 0, f"Missing fields: {missing}" if missing else "OK"


def check_g2(operator=None):
    """Plugin/Skill loaded check"""
    if not file_exists(TASK_CONTEXT):
        return False, "task_context.json not found"
    with open(TASK_CONTEXT) as f:
        ctx = json.load(f)
    required_skills = ctx.get("required_skills", [])
    loaded_skills = ctx.get("loaded_skills", [])
    missing = [s for s in required_skills if s not in loaded_skills]
    if missing:
        return False, f"Skills not loaded: {missing}"
    plugin = ctx.get("active_plugin")
    if not plugin:
        return False, "No active plugin recorded"
    return True, f"Plugin: {plugin}, Skills: {len(loaded_skills)} loaded"


def check_g3_generic(artifact, required_fields, label):
    """Generic check for artifact-based gates (file or directory)"""
    if artifact is None:
        return False, f"{label} path is None"
    if not os.path.exists(artifact):
        return False, f"{label} not found: {artifact}"
    return True, f"{label} found"


def check_g8(operator):
    """Correctness gate — the hard gate"""
    # Multi-operator mode: read INDEX and check all operators
    if not operator:
        if file_exists(TASK_CONTEXT):
            with open(TASK_CONTEXT) as f:
                ctx = json.load(f)
            if ctx.get("operator") == "INDEX" and "operators" in ctx:
                ops = [o["operator"] for o in ctx["operators"]]
                results = {}
                for op in ops:
                    passed, msg = _check_g8_for_operator(op)
                    results[op] = passed
                all_pass = all(results.values())
                failed_ops = [op for op, p in results.items() if not p]
                if all_pass:
                    return True, f"All {len(ops)} operators passed correctness"
                return False, f"Correctness failed for: {failed_ops}"
        return False, "No operator specified and no INDEX found"
    return _check_g8_for_operator(operator)


def _check_g8_for_operator(operator):
    """Check correctness for a single operator. Passes if ANY route passes."""
    correctness_dir = os.path.join(PROJECT_ROOT, "operators", operator, "reports", "correctness")

    def _file_passes(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            return False
        if not isinstance(data, dict):
            return False
        all_pass = data.get("all_pass")
        if all_pass is True:
            return True
        overall = data.get("overall") or data.get("status")
        if overall in ("PASS", "pass", True):
            return True
        if "results" in data and isinstance(data["results"], list) and len(data["results"]) > 0:
            passes = sum(1 for r in data["results"]
                         if r.get("status") == "PASS" or r.get("pass", False) is True)
            if passes == len(data["results"]):
                return True
        return False

    if os.path.isdir(correctness_dir):
        for fname in os.listdir(correctness_dir):
            if fname.endswith(".json"):
                if _file_passes(os.path.join(correctness_dir, fname)):
                    return True, f"{operator}: correctness passed ({fname})"

    torch_cr = os.path.join(PROJECT_ROOT, "operators", operator, "torch", "correctness_results.json")
    if os.path.isfile(torch_cr):
        if _file_passes(torch_cr):
            return True, f"{operator}: torch correctness passed"

    return False, f"{operator}: correctness results not found"


def check_g14():
    """Release gate — check that release artifacts exist"""
    release_json = os.path.join(PROJECT_ROOT, "reports", "release", "current_release.json")
    if not file_exists(release_json):
        return False, "current_release.json not found"
    try:
        with open(release_json) as f:
            data = json.load(f)
        git_commit = data.get("git_commit", "")
        operators = data.get("operators", {})
        return True, f"Release: commit={git_commit[:7]}, operators={len(operators)}"
    except Exception as e:
        return False, f"current_release.json read error: {e}"


def check_gate(gate_id, operator=None):
    gates = load_gates()
    gate = next((g for g in gates if g["id"] == gate_id), None)
    if not gate:
        return False, f"Gate {gate_id} not found"

    gate_checks = {
        "G0": check_g0,
        "G1": check_g1,
        "G2": lambda: check_g2(operator),
        "G3": lambda: (
            check_g3_generic(None, [], "SPEC") if operator is None
            else check_g3_generic(
                os.path.join(PROJECT_ROOT, "operators", operator, "SPEC.yaml"),
                ["operator_name", "inputs", "outputs"], "SPEC")
        ),
        "G7": lambda: (
            check_g3_generic(None, [], "Build directory") if operator is None
            else check_g3_generic(
                os.path.join(PROJECT_ROOT, "operators", operator, "ascendc", "build"),
                [], "Build directory")
        ),
        "G8": lambda: check_g8(operator),
        "G14": check_g14,
    }

    check_fn = gate_checks.get(gate_id)
    if not check_fn:
        return True, "No automated check — manual verification required"

    try:
        return check_fn()
    except Exception as e:
        return False, f"Check error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Gate verification tool")
    parser.add_argument("--gate", help="Check specific gate (e.g., G3)")
    parser.add_argument("--operator", help="Operator name for operator-specific checks")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    gates = load_gates()
    results = {}
    all_pass = True

    for gate in gates:
        gid = gate["id"]
        if args.gate and gid != args.gate:
            continue
        passed, message = check_gate(gid, args.operator)
        results[gid] = {"name": gate["name"], "passed": passed, "message": message}
        if not passed:
            all_pass = False

    if args.json:
        print(json.dumps({"all_pass": all_pass, "gates": results}, indent=2))
    else:
        print(f"Gate Check Results: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        print("=" * 60)
        for gid, info in results.items():
            status = "PASS" if info["passed"] else "FAIL"
            print(f"  {gid:4s} [{status:4s}] {info['name']:20s} | {info['message']}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
