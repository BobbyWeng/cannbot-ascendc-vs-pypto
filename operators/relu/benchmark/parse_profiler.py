#!/usr/bin/env python3
"""ReLU profiler parser — delegates to common/profiler/parse_profiler_v2.py"""
import os, sys
_common_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'common', 'profiler')
sys.path.insert(0, _common_dir)
from parse_profiler_v2 import main
if __name__ == '__main__':
    main()
