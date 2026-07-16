import os, sys
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from correctness import check_correctness
sys.path.remove(_dir)

__all__ = ["check_correctness"]
