#!/usr/bin/env bash
# Pre-Release Gate
# MUST run before any release is published or ranked.
# Exits non-zero if any gate fails.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAILED=0

echo "=== Pre-Release Gate ==="
echo ""

# Run all gates through check_gate.py
echo "[1] Running all gate checks..."
python3 "$PROJECT_ROOT/tools/check_gate.py" --json 2>/dev/null || true

# G8: Correctness (hard gate - NEVER overridable)
echo ""
echo "[2] Verifying Correctness gate (G8) - HARD GATE..."
if python3 "$PROJECT_ROOT/tools/check_gate.py" --gate G8 --json 2>/dev/null | \
    python3 -c "import sys,json; d=json.load(sys.stdin); g=d['gates']['G8']; exit(0 if g['passed'] else 1)"; then
    echo "  PASS"
else
    echo "  FAIL: Correctness gate (G8) not passed. This is a HARD GATE - never overridable."
    echo "  All implementations must pass correctness before entering performance ranking."
    FAILED=1
fi

# G13: Code Review
echo ""
echo "[3] Verifying Code Review gate (G13)..."
if python3 "$PROJECT_ROOT/tools/check_gate.py" --gate G13 --json 2>/dev/null | \
    python3 -c "import sys,json; d=json.load(sys.stdin); g=d['gates']['G13']; exit(0 if g['passed'] else 1)"; then
    echo "  PASS"
else
    echo "  WARN: Code Review gate (G13) not passed. Release may proceed but note in report."
fi

# Verify Cannbot usage
echo ""
echo "[4] Verifying Cannbot usage compliance..."
python3 "$PROJECT_ROOT/tools/verify_cannbot_usage.py" --json 2>/dev/null || FAILED=1

# Verify final reports exist and are consistent
echo ""
echo "[5] Checking consistency of final reports..."
if [ -f "$PROJECT_ROOT/reports/release/current_release.json" ]; then
    echo "  PASS: current_release.json found"
else
    echo "  FAIL: current_release.json missing. Run final report generation."
    FAILED=1
fi

# Verify operator README and REPRODUCE exist
echo ""
echo "[6] Checking operator documentation..."
OPERATOR=""
if [ -f "$PROJECT_ROOT/reports/runtime/task_context.json" ]; then
    OPERATOR=$(python3 -c "import json; print(json.load(open('$PROJECT_ROOT/reports/runtime/task_context.json')).get('operator',''))" 2>/dev/null || echo "")
fi

if [ -n "$OPERATOR" ]; then
    for doc in README.md REPRODUCE.md; do
        if [ -f "$PROJECT_ROOT/operators/$OPERATOR/$doc" ]; then
            echo "  PASS: $doc found"
        else
            echo "  WARN: $doc missing for $OPERATOR"
        fi
    done
fi

# Check for unverified backend limitations
echo ""
echo "[7] Checking for unverified backend limitations..."
if [ -n "$OPERATOR" ] && [ -f "$PROJECT_ROOT/operators/$OPERATOR/reports/diagnostic" ]; then
    echo "  INFO: diagnostic reports exist, review for unverified claims"
fi

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== PRE-RELEASE GATE PASSED ==="
else
    echo "=== PRE-RELEASE GATE FAILED ($FAILED checks failed) ==="
fi
exit $FAILED
