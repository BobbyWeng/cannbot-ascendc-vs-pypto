#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

show_help() {
  echo "Usage: $0 --type={parser|kernel|dashboard|correctness|all}"
  echo ""
  echo "Modes:"
  echo "  --type=parser       Run SHA256 + parsed check (parser code changed)"
  echo "  --type=kernel       Run SHA256 + build + profiler check (kernel code changed)"
  echo "  --type=dashboard    Run dashboard validation only"
  echo "  --type=correctness  Run SHA256 + build + correctness check"
  echo "  --type=all          Run full regression suite"
  exit 0
}

if [ $# -eq 0 ]; then
  show_help
fi

CHANGE_TYPE=""
for arg in "$@"; do
  case "$arg" in
    --type=parser)     CHANGE_TYPE="parser" ;;
    --type=kernel)     CHANGE_TYPE="kernel" ;;
    --type=dashboard)  CHANGE_TYPE="dashboard" ;;
    --type=correctness) CHANGE_TYPE="correctness" ;;
    --type=all)        CHANGE_TYPE="all" ;;
    --help)            show_help ;;
    *)
      echo "Unknown option: $arg"
      show_help
      ;;
  esac
done

if [ -z "$CHANGE_TYPE" ]; then
  show_help
fi

echo "=================================================================="
echo "  Cannbot Post-Change Regression — type=$CHANGE_TYPE"
echo "  Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "=================================================================="
echo ""

case "$CHANGE_TYPE" in
  parser)
    echo "Running checks for parser changes..."
    echo ""
    "$SCRIPT_DIR/run_regression.sh" --skip-build --skip-correctness --skip-profiler
    ;;

  kernel)
    echo "Running checks for kernel changes..."
    echo ""
    "$SCRIPT_DIR/run_regression.sh" --skip-correctness --skip-parsed
    ;;

  dashboard)
    echo "Running dashboard validation..."
    echo ""
    DASHBOARD="$PROJECT_ROOT/dashboard/dashboard.json"
    if python3 "$SCRIPT_DIR/check_dashboard.py" "$DASHBOARD"; then
      echo ""
      echo "DASHBOARD VALIDATION PASSED"
      exit 0
    else
      echo ""
      echo "DASHBOARD VALIDATION FAILED"
      exit 1
    fi
    ;;

  correctness)
    echo "Running checks for correctness changes..."
    echo ""
    "$SCRIPT_DIR/run_regression.sh" --skip-profiler --skip-parsed
    ;;

  all)
    echo "Running full regression suite..."
    echo ""
    "$SCRIPT_DIR/run_regression.sh"
    ;;
esac
