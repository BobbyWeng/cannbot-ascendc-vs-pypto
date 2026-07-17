#!/usr/bin/env bash
# Pre-Kernel-Commit Gate
# MUST run before any kernel source is committed.
# Exits non-zero if any gate fails.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAILED=0

echo "=== Pre-Kernel-Commit Gate ==="
echo ""

# 1. Verify task_context exists
echo "[1/7] Checking task_context.json..."
if [ ! -f "$PROJECT_ROOT/reports/runtime/task_context.json" ]; then
    echo "  FAIL: task_context.json missing. Run task classification first."
    FAILED=1
else
    echo "  PASS"
fi

# 2. Verify Cannbot usage
echo "[2/7] Checking Cannbot usage..."
if python3 "$PROJECT_ROOT/tools/verify_cannbot_usage.py" --stage check; then
    echo "  PASS"
else
    FAILED=1
fi

# 3. Check SKILL_TRACE exists
echo "[3/7] Checking SKILL_TRACE..."
OPERATOR=""
if [ -f "$PROJECT_ROOT/reports/runtime/task_context.json" ]; then
    OPERATOR=$(python3 -c "import json; print(json.load(open('$PROJECT_ROOT/reports/runtime/task_context.json')).get('operator',''))" 2>/dev/null || echo "")
fi

if [ -n "$OPERATOR" ] && [ -f "$PROJECT_ROOT/operators/$OPERATOR/SKILL_TRACE.json" ]; then
    echo "  PASS (SKILL_TRACE.json found for $OPERATOR)"
elif [ -f "$PROJECT_ROOT/SKILL_TRACE.json" ]; then
    echo "  PASS (SKILL_TRACE.json found at project root)"
else
    echo "  WARN: SKILL_TRACE.json not found"
fi

# 4. Check gate G8 (correctness)
echo "[4/7] Checking correctness gate (G8)..."
if python3 "$PROJECT_ROOT/tools/check_gate.py" --gate G8 --operator "$OPERATOR" --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d['gates']['G8']['passed'] else 1)" 2>/dev/null; then
    echo "  PASS"
else
    echo "  FAIL: Correctness not verified. Run correctness checks before committing kernel code."
    FAILED=1
fi

# 5. Check for profiler data if optimization was performed
echo "[5/7] Checking profiler data..."
TASK_MODE=$(python3 -c "import json; t=json.load(open('$PROJECT_ROOT/reports/runtime/task_context.json')); print(t.get('task_mode',''))" 2>/dev/null || echo "")
if [ "$TASK_MODE" = "PERFORMANCE_OPTIMIZATION" ]; then
    if python3 "$PROJECT_ROOT/tools/check_gate.py" --gate G12 --operator "$OPERATOR" --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d['gates']['G12']['passed'] else 1)" 2>/dev/null; then
        echo "  PASS"
    else
        echo "  FAIL: Final profile missing for optimization task."
        FAILED=1
    fi
else
    echo "  SKIP (task is not PERFORMANCE_OPTIMIZATION)"
fi

# 6. Check Git diff
echo "[6/7] Checking uncommitted changes..."
if git -C "$PROJECT_ROOT" diff --quiet 2>/dev/null; then
    echo "  PASS (no uncommitted changes)"
else
    echo "  INFO: Uncommitted changes detected:"
    git -C "$PROJECT_ROOT" diff --stat
fi

# 7. Check SHA256SUMS
echo "[7/7] Checking SHA256SUMS..."
if [ -n "$OPERATOR" ] && [ -f "$PROJECT_ROOT/operators/$OPERATOR/SHA256SUMS" ]; then
    if (cd "$PROJECT_ROOT/operators/$OPERATOR" && sha256sum -c SHA256SUMS 2>/dev/null); then
        echo "  PASS"
    else
        echo "  WARN: SHA256SUMS verification failed (may be expected with new changes)"
    fi
else
    echo "  SKIP (no SHA256SUMS for operator)"
fi

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== PRE-KERNEL-COMMIT GATE PASSED ==="
else
    echo "=== PRE-KERNEL-COMMIT GATE FAILED ($FAILED checks failed) ==="
fi
exit $FAILED
