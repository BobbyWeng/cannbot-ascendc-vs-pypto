import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OPERATORS = [
    "add", "div", "equal", "expand", "matmul", "mul",
    "not", "or", "reduce_sum", "relu", "transpose", "where",
]

OPERATOR_DIR = PROJECT_ROOT / "operators"
REPORTS_DIR = PROJECT_ROOT / "reports"
RELEASE_DIR = REPORTS_DIR / "release"
FINAL_DIR = REPORTS_DIR / "final"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
REGRESSION_DIR = PROJECT_ROOT / "tests" / "regression"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TOOLS_DIR = PROJECT_ROOT / "tools"

RELEASE_MANIFEST = RELEASE_DIR / "release_manifest.json"
CURRENT_RELEASE = RELEASE_DIR / "current_release.json"
LIMITATION_MATRIX = RELEASE_DIR / "limitation_matrix.json"
CORRECTNESS_MATRIX = RELEASE_DIR / "correctness_matrix.csv"
PERFORMANCE_MATRIX = RELEASE_DIR / "performance_matrix.csv"
OPERATOR_MATRIX = RELEASE_DIR / "operator_matrix.csv"
CURRENT_RELEASE_MD = RELEASE_DIR / "current_release.md"
FINAL_COMPARISON = FINAL_DIR / "final_comparison.md"
FINAL_COMPARISON_JSON = FINAL_DIR / "final_comparison.json"
FINAL_COMPARISON_CSV = FINAL_DIR / "final_comparison.csv"
RELEASE_CHANGELOG = PROJECT_ROOT / "RELEASE_CHANGELOG.md"
DASHBOARD_JSON = DASHBOARD_DIR / "dashboard.json"
DASHBOARD_HTML = DASHBOARD_DIR / "index.html"
SHA256SUMS_FILE = "SHA256SUMS"

RELEASE_VERSION = "1.2-rc3"
DEFAULT_ENV = {
    "platform": "Ascend 910B",
    "cann_version": "9.0.0",
    "python": "3.11",
    "profiler": "msprof with --ascendcl=on --ai-core=on --task-time=l0",
    "warmup": 200,
    "profiled_loops": 100,
    "repeat": "5",
    "primary_metric": "primary_compute_kernel_us",
}
