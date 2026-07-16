import os, sys
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from benchmark_utils import compute_statistics, compute_effective_bandwidth
sys.path.remove(_dir)

__all__ = ["compute_statistics", "compute_effective_bandwidth"]
