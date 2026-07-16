#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 5 ]; then
    echo "Usage: $0 <operator> <implementation> <batch> <stage> <command...>"
    exit 1
fi

OPERATOR=$1
IMPLEMENTATION=$2
BATCH=$3
STAGE=$4
shift 4

bash scripts/acquire_npu_lock.sh "$OPERATOR" "$IMPLEMENTATION" "$BATCH" "$STAGE" 0
RET=$?
if [ $RET -ne 0 ]; then
    echo "NPU_LOCK_FAILED operator=$OPERATOR batch=$BATCH stage=$STAGE"
    exit $RET
fi

echo "NPU_RUN_START operator=$OPERATOR impl=$IMPLEMENTATION batch=$BATCH stage=$STAGE pid=$$"
set +e
eval "$@"
CMD_RET=$?
set -e

bash scripts/release_npu_lock.sh "$OPERATOR"
echo "NPU_RUN_END operator=$OPERATOR impl=$IMPLEMENTATION batch=$BATCH stage=$STAGE ret=$CMD_RET"
exit $CMD_RET
