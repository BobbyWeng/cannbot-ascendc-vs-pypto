#!/bin/bash
LOCK_FILE="reports/runtime/npu.lock"
if [ -f "$LOCK_FILE" ]; then
    echo "LOCK_RELEASED operator=$1 pid=$(python3 -c "import json; print(json.load(open('$LOCK_FILE')).get('pid', 'unknown'))" 2>/dev/null || echo unknown)"
    rm -f "$LOCK_FILE"
fi
