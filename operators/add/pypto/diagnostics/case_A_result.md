# Case A Result: FP16 [1,32] + [1,32] Binary Add

## Summary
- **Status**: ✅ PASS
- **Bitwise equal**: True
- **Exit code**: 0

## Source
- `add_impl.py`: kernel defined in imported module (not in `__main__`)
- Uses `pypto.op.add(x1, x2)` API (returns new tensor, stored via `y.move(...)`)

## Key Findings
1. `pypto.op.add(input, other, *, alpha=1) -> Tensor` **works** — this is the correct API
2. The earlier failures were caused by:
   - Using `+` operator overload (`x1 + x2`) instead of `pypto.op.add(x1, x2)` — the `+` operator may parse differently in the frontend
   - Defining JIT functions in `__main__` (failed with "function nested is not allowed")

## Commands
```bash
source /etc/profile.d/ascend_env.sh
source ~/Ascend/ascend-toolkit/set_env.sh
TILE_FWK_DEVICE_ID=0 python3 case_a.py
```

## Version
- pypto: /home/developer/.local/lib/python3.11/site-packages/pypto/
- torch_npu: installed
- Device: Ascend 910B
