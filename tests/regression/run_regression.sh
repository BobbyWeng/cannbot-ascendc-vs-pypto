#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ALL_OPERATORS=(add div equal expand matmul mul not or reduce_sum relu transpose where softmax layernorm)

SKIP_SHA256=false
SKIP_BUILD=false
SKIP_CORRECTNESS=false
SKIP_PROFILER=false
SKIP_PARSED=false
JSON_OUTPUT=false

for arg in "$@"; do
  case "$arg" in
    --skip-sha256) SKIP_SHA256=true ;;
    --skip-build) SKIP_BUILD=true ;;
    --skip-correctness) SKIP_CORRECTNESS=true ;;
    --skip-profiler) SKIP_PROFILER=true ;;
    --skip-parsed) SKIP_PARSED=true ;;
    --json) JSON_OUTPUT=true ;;
    --help)
      echo "Usage: $0 [--skip-sha256] [--skip-build] [--skip-correctness] [--skip-profiler] [--skip-parsed] [--json]"
      echo ""
      echo "Checks:"
      echo "  sha256     Verify SHA256SUMS for each operator"
      echo "  build      Check ascendc build binary exists"
      echo "  correctness Check torch correctness_results.json for PASS status"
      echo "  profiler   Verify parsed profiler JSON files exist"
      echo "  parsed     Validate parsed JSON file content (field presence/constraints)"
      echo "  --json     Output machine-readable JSON summary"
      exit 0 ;;
    *)
      echo "Unknown option: $arg"
      exit 1 ;;
  esac
done

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
OPERATOR_RESULTS=""

run_check() {
  local op="$1"
  local check_name="$2"
  local status="$3"
  local detail="$4"
  OPERATOR_RESULTS="${OPERATOR_RESULTS}  [${status}] ${op}: ${check_name}${detail:+ — ${detail}}"$'\n'
  if [ "$status" = "PASS" ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

run_skip() {
  local op="$1"
  local check_name="$2"
  OPERATOR_RESULTS="${OPERATOR_RESULTS}  [SKIP] ${op}: ${check_name}"$'\n'
  SKIP_COUNT=$((SKIP_COUNT + 1))
}

check_sha256() {
  local op="$1"
  local op_dir="$PROJECT_ROOT/operators/$op"
  local sha_file="$op_dir/SHA256SUMS"
  if [ ! -f "$sha_file" ]; then
    run_check "$op" "sha256" "FAIL" "SHA256SUMS file not found"
    return
  fi
  if cd "$op_dir" 2>/dev/null && sha256sum -c "$sha_file" > /dev/null 2>&1; then
    run_check "$op" "sha256" "PASS"
  else
    run_check "$op" "sha256" "FAIL" "checksum mismatch"
  fi
}

check_build() {
  local op="$1"
  local build_path="$PROJECT_ROOT/operators/$op/ascendc/build/${op}_ascendc"
  if [ -f "$build_path" ] || [ -d "$build_path" ]; then
    run_check "$op" "build" "PASS"
  else
    run_check "$op" "build" "FAIL" "build artifact not found at $build_path"
  fi
}

check_correctness() {
  local op="$1"
  local torch_result="$PROJECT_ROOT/operators/$op/torch/correctness_results.json"
  if [ -f "$torch_result" ]; then
    if python3 -c "
import json, sys
with open('$torch_result') as f:
    data = json.load(f)
results = data.get('results', [])
if not results:
    results = data if isinstance(data, list) else [data]
failures = [r for r in results if r.get('status') != 'PASS']
if failures:
    sys.exit(1)
" 2>/dev/null; then
      run_check "$op" "correctness/torch" "PASS"
    else
      run_check "$op" "correctness/torch" "FAIL" "some torch batches not PASS"
    fi
  else
    run_skip "$op" "correctness/torch"
  fi

  local ascendc_correctness="$PROJECT_ROOT/operators/$op/reports/correctness"
  if [ -d "$ascendc_correctness" ] && [ "$(ls -A "$ascendc_correctness" 2>/dev/null)" ]; then
    run_check "$op" "correctness/ascendc" "PASS"
  else
    run_skip "$op" "correctness/ascendc"
  fi
}

check_profiler() {
  local op="$1"
  local parsed_dir="$PROJECT_ROOT/operators/$op/reports/parsed"
  if [ ! -d "$parsed_dir" ]; then
    run_skip "$op" "profiler" "no parsed dir"
    return
  fi
  local files
  files=$(ls "$parsed_dir"/*.json 2>/dev/null || true)
  if [ -z "$files" ]; then
    run_check "$op" "profiler" "FAIL" "no parsed JSON files"
    return
  fi
  run_check "$op" "profiler" "PASS" "$(echo "$files" | wc -l) parsed files"
}

check_parsed() {
  local op="$1"
  local parsed_dir="$PROJECT_ROOT/operators/$op/reports/parsed"
  if [ ! -d "$parsed_dir" ]; then
    run_skip "$op" "parsed-validation" "no parsed dir"
    return
  fi
  local any_fail=false
  local count=0
  for f in "$parsed_dir"/*.json; do
    [ -f "$f" ] || continue
    local fname
    fname=$(basename "$f")
    # skip aggregate.json — it uses a different aggregated schema
    [ "$fname" = "aggregate.json" ] && continue
    count=$((count + 1))
    if ! python3 "$SCRIPT_DIR/check_parsed.py" "$f" > /dev/null 2>&1; then
      any_fail=true
    fi
  done
  if [ "$count" -eq 0 ]; then
    run_skip "$op" "parsed-validation" "no JSON files (excluding aggregate.json)"
  elif $any_fail; then
    run_check "$op" "parsed-validation" "FAIL" "some parsed files failed validation"
  else
    run_check "$op" "parsed-validation" "PASS" "$count files validated"
  fi
}

generate_json_summary() {
  local json_data
  json_data=$(cat <<JSONEOF
{
  "suite": "cannbot-regression",
  "project": "cannbot_ascendc_vs_pypto",
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "total_operators": ${#ALL_OPERATORS[@]},
  "results": {
    "passed": $PASS_COUNT,
    "failed": $FAIL_COUNT,
    "skipped": $SKIP_COUNT
  }
}
JSONEOF
)
  echo "$json_data"
}

echo "=================================================================="
echo "  Cannbot Regression Test Suite"
echo "  Project: cannbot_ascendc_vs_pypto"
echo "  Date:    $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Operators: ${ALL_OPERATORS[*]}"
echo "=================================================================="
echo ""

for op in "${ALL_OPERATORS[@]}"; do
  echo "--- Operator: $op ---"

  $SKIP_SHA256      || check_sha256 "$op"
  $SKIP_BUILD       || check_build "$op"
  $SKIP_CORRECTNESS || check_correctness "$op"
  $SKIP_PROFILER    || check_profiler "$op"
  $SKIP_PARSED      || check_parsed "$op"

  echo ""
done

if $JSON_OUTPUT; then
  generate_json_summary
else
  echo "=== Final Report ==="
  echo ""
  echo "$OPERATOR_RESULTS" | column -t -s $'\t' 2>/dev/null || echo "$OPERATOR_RESULTS"
  echo ""
  echo "Summary: ${PASS_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped"
  echo ""
fi

if [ "$FAIL_COUNT" -eq 0 ]; then
  [ "$JSON_OUTPUT" = false ] && echo "ALL CHECKS PASSED"
  exit 0
else
  [ "$JSON_OUTPUT" = false ] && echo "SOME CHECKS FAILED"
  exit 1
fi
