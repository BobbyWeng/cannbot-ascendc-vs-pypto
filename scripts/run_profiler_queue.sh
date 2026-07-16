#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

QUEUE_FILE="reports/batches/logical_ops_v1/npu_run_queue.json"
mkdir -p reports/batches/logical_ops_v1
mkdir -p reports/runtime

# Define the queue
cat > "$QUEUE_FILE" << 'QUEUE_EOF'
{
  "queue": [
    {"op": "equal", "impl": "ascendc", "batch": 1},
    {"op": "equal", "impl": "ascendc", "batch": 2},
    {"op": "equal", "impl": "ascendc", "batch": 4},
    {"op": "equal", "impl": "ascendc", "batch": 8},
    {"op": "equal", "impl": "ascendc", "batch": 16},
    {"op": "equal", "impl": "ascendc", "batch": 32},
    {"op": "equal", "impl": "ascendc", "batch": 64},
    {"op": "equal", "impl": "torch", "batch": 1},
    {"op": "equal", "impl": "torch", "batch": 2},
    {"op": "equal", "impl": "torch", "batch": 4},
    {"op": "equal", "impl": "torch", "batch": 8},
    {"op": "equal", "impl": "torch", "batch": 16},
    {"op": "equal", "impl": "torch", "batch": 32},
    {"op": "equal", "impl": "torch", "batch": 64},
    {"op": "where", "impl": "ascendc", "batch": 1},
    {"op": "where", "impl": "ascendc", "batch": 2},
    {"op": "where", "impl": "ascendc", "batch": 4},
    {"op": "where", "impl": "ascendc", "batch": 8},
    {"op": "where", "impl": "ascendc", "batch": 16},
    {"op": "where", "impl": "ascendc", "batch": 32},
    {"op": "where", "impl": "ascendc", "batch": 64},
    {"op": "where", "impl": "torch", "batch": 1},
    {"op": "where", "impl": "torch", "batch": 2},
    {"op": "where", "impl": "torch", "batch": 4},
    {"op": "where", "impl": "torch", "batch": 8},
    {"op": "where", "impl": "torch", "batch": 16},
    {"op": "where", "impl": "torch", "batch": 32},
    {"op": "where", "impl": "torch", "batch": 64},
    {"op": "not", "impl": "torch", "batch": 1},
    {"op": "not", "impl": "torch", "batch": 2},
    {"op": "not", "impl": "torch", "batch": 4},
    {"op": "not", "impl": "torch", "batch": 8},
    {"op": "not", "impl": "torch", "batch": 16},
    {"op": "not", "impl": "torch", "batch": 32},
    {"op": "not", "impl": "torch", "batch": 64},
    {"op": "not", "impl": "ascendc", "batch": 1},
    {"op": "not", "impl": "ascendc", "batch": 2},
    {"op": "not", "impl": "ascendc", "batch": 4},
    {"op": "not", "impl": "ascendc", "batch": 8},
    {"op": "not", "impl": "ascendc", "batch": 16},
    {"op": "not", "impl": "ascendc", "batch": 32},
    {"op": "not", "impl": "ascendc", "batch": 64},
    {"op": "not", "impl": "pypto", "batch": 1},
    {"op": "not", "impl": "pypto", "batch": 2},
    {"op": "not", "impl": "pypto", "batch": 4},
    {"op": "not", "impl": "pypto", "batch": 8},
    {"op": "not", "impl": "pypto", "batch": 16},
    {"op": "not", "impl": "pypto", "batch": 32},
    {"op": "not", "impl": "pypto", "batch": 64},
    {"op": "or", "impl": "torch", "batch": 1},
    {"op": "or", "impl": "torch", "batch": 2},
    {"op": "or", "impl": "torch", "batch": 4},
    {"op": "or", "impl": "torch", "batch": 8},
    {"op": "or", "impl": "torch", "batch": 16},
    {"op": "or", "impl": "torch", "batch": 32},
    {"op": "or", "impl": "torch", "batch": 64},
    {"op": "or", "impl": "ascendc", "batch": 1},
    {"op": "or", "impl": "ascendc", "batch": 2},
    {"op": "or", "impl": "ascendc", "batch": 4},
    {"op": "or", "impl": "ascendc", "batch": 8},
    {"op": "or", "impl": "ascendc", "batch": 16},
    {"op": "or", "impl": "ascendc", "batch": 32},
    {"op": "or", "impl": "ascendc", "batch": 64},
    {"op": "or", "impl": "pypto", "batch": 1},
    {"op": "or", "impl": "pypto", "batch": 2},
    {"op": "or", "impl": "pypto", "batch": 4},
    {"op": "or", "impl": "pypto", "batch": 8},
    {"op": "or", "impl": "pypto", "batch": 16},
    {"op": "or", "impl": "pypto", "batch": 32},
    {"op": "or", "impl": "pypto", "batch": 64}
  ]
}
QUEUE_EOF

echo "NPU queue created. Total items: $(python3 -c "import json; q=json.load(open('$QUEUE_FILE')); print(len(q['queue']))")"

# Process queue
python3 -c "
import json, subprocess, sys, os

queue_file = '$QUEUE_FILE'
project_root = '$PROJECT_ROOT'
queue = json.load(open(queue_file))

results = []
for i, item in enumerate(queue['queue']):
    op = item['op']
    impl = item['impl']
    batch = item['batch']
    
    print(f'\\n=== QUEUE [{i+1}/{len(queue[\"queue\"])}] {op} {impl} B={batch} ===')
    
    if impl == 'ascendc':
        # Ascend C: use built-in binary
        binary = f'{project_root}/operators/{op}/ascendc/build/{op}_ascendc'
        data_dir = f'{project_root}/operators/{op}/data'
        output_dir = f'{project_root}/operators/{op}/ascendc/build/output'
        raw_dir = f'{project_root}/operators/{op}/reports/raw/{impl}/b{batch}'
        os.makedirs(raw_dir, exist_ok=True)
        
        cmd = [binary, '0', str(batch), '20', '8192', '200', '100', '5', data_dir, output_dir]
        subprocess.run(cmd, check=True)
        
        # Run again through lock
    elif impl == 'torch':
        cmd = ['python3', f'{project_root}/common/profiler/run_profiler.py', op, impl, str(batch)]
        subprocess.run(cmd, check=True)
    elif impl == 'pypto':
        cmd = ['python3', f'{project_root}/common/profiler/run_profiler.py', op, impl, str(batch)]
        subprocess.run(cmd, check=True)
    
    results.append({**item, 'status': 'done'})
    with open(queue_file.replace('.json', '_progress.json'), 'w') as f:
        json.dump({'completed': i+1, 'total': len(queue['queue']), 'results': results}, f, indent=2)

print(f'\\n=== ALL DONE: {len(results)} items ===')
" 2>&1
