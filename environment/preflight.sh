#!/bin/bash
# Preflight check script for cannbot_ascendc_vs_pypto
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Preflight Check ==="
echo ""

# Python version
echo -n "Python: "
python3 --version 2>&1 || echo "NOT FOUND"

# CANN
echo -n "ASCEND_HOME_PATH: "
if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    echo "$ASCEND_HOME_PATH"
    if [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
        source "$ASCEND_HOME_PATH/set_env.sh"
    fi
else
    echo "NOT SET"
fi

# CANN Toolkit version
echo -n "CANN Toolkit: "
if command -v msprof &> /dev/null; then
    msprof --version 2>/dev/null | head -1 || echo "msprof available (no version flag)"
else
    echo "msprof NOT FOUND (try: source set_env.sh)"
fi

# NPU device
echo -n "NPU devices: "
if command -v npu-smi &> /dev/null; then
    npu-smi info 2>/dev/null | head -10 || echo "npu-smi failed"
else
    echo "npu-smi NOT FOUND"
fi

# PyTorch
echo -n "torch: "
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT FOUND"
echo -n "torch_npu: "
python3 -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null || echo "NOT FOUND"

# PyPTO
echo -n "pypto: "
python3 -c "import pypto; print(pypto.__version__)" 2>/dev/null || echo "pypto NOT FOUND"

# CMake
echo -n "cmake: "
cmake --version 2>/dev/null | head -1 || echo "NOT FOUND"
echo -n "make: "
make --version 2>/dev/null | head -1 || echo "NOT FOUND"

# Device count
echo -n "NPU count: "
python3 -c "
import torch, torch_npu
count = torch.npu.device_count()
print(f'{count} device(s)')
for i in range(count):
    print(f'  device {i}: {torch.npu.get_device_name(i)}')
" 2>/dev/null || echo "Failed to query NPU"

echo ""
echo "=== Done ==="
