# Verified PyPTO JIT Template

A minimal, tested template for PyPTO JIT kernel development.

## Structure

```
verified_jit_template/
├── src/
│   └── relu_impl.py           # JIT function + wrapper (top-level only)
├── tests/
│   └── test_relu.py           # Source introspection + NPU correctness
├── golden/
│   └── relu_golden.py         # Pure PyTorch reference
├── scripts/
│   └── inspect_source.py      # Pre-flight source check utility
├── run_jit.sh                 # Full test runner
└── README.md
```

## Rules (MUST follow)

1. JIT function defined ONLY in `src/{op}_impl.py` — top-level, not nested
2. Import via `sys.path.insert` + direct module name (not package-qualified)
3. Wrapper function in same file, NOT decorated with @jit
4. Test imports wrapper (not the JIT function directly)
5. No lambda, no closure, no heredoc, no `__main__` JIT definition
6. `__init__.py` is OPTIONAL — works without it
7. Source file must exist on disk at call time

## Verification

```bash
cd /mnt/workspace/cannbot_ascendc_vs_pypto
bash common/pypto/verified_jit_template/run_jit.sh
```
