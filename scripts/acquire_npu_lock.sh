#!/bin/bash
set -e
LOCK_DIR="reports/runtime"
LOCK_FILE="$LOCK_DIR/npu.lock"
mkdir -p "$LOCK_DIR"

attempt=0
max_attempts=60
sleep_sec=2

while [ $attempt -lt $max_attempts ]; do
    if mkdir "$LOCK_FILE.lockdir" 2>/dev/null; then
        cat > "$LOCK_FILE" <<EOF
{
  "operator": "$1",
  "implementation": "$2",
  "batch": "$3",
  "stage": "$4",
  "pid": $$,
  "device": "${5:-0}",
  "hostname": "$(hostname)",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "command": "$@"
}
EOF
        rmdir "$LOCK_FILE.lockdir" 2>/dev/null || true
        echo "LOCK_ACQUIRED operator=$1 pid=$$"
        exit 0
    fi
    # Check for stale lock (PID no longer exists)
    if [ -f "$LOCK_FILE" ]; then
        LOCKED_PID=$(python3 -c "import json; print(json.load(open('$LOCK_FILE')).get('pid', 0))" 2>/dev/null || echo 0)
        if [ "$LOCKED_PID" -gt 0 ] && ! kill -0 "$LOCKED_PID" 2>/dev/null; then
            echo "WARNING: Stale lock from PID $LOCKED_PID, clearing"
            rm -f "$LOCK_FILE"
            continue
        fi
    fi
    attempt=$((attempt + 1))
    echo "LOCK_WAITING attempt=$attempt/$max_attempts operator=$1"
    sleep $sleep_sec
done

echo "LOCK_FAILED operator=$1 after $max_attempts attempts"
exit 1
