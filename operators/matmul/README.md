# MatMul Operator — First Cube Operator

## Status: COMPLETE_WITH_LIMITATION

- **Torch**: ✅ PASS (all 6 batches, atol=0.03125, rtol=1.5)
- **Ascend C**: ⚠️ TRUE_DEVICE_AIVEC (Vector path with FP32 accumulation, NOT Cube)
- **PyPTO**: ❌ BLOCKED_BACKEND (backend CompileFunction error)

This is the first Cube-class operator in the project. It compares three routes:
1. **Torch** — `torch.matmul` baseline (uses `aclnnMatmul_BatchMatMulNd_BatchMatMulV2` Cube path)
2. **Ascend C** — AIC Vector MatMul (scalar FP32 accumulation)
3. **PyPTO** — `pypto.frontend.jit` with `pypto.matmul` (BLOCKED_BACKEND)

## Shape
- A: [B, 12, 256, 256], FP16
- B: [B, 12, 256, 32], FP16
- Y: [B, 12, 256, 32], FP16
- B: 1, 2, 4, 8, 16, 32

## Files
- `SPEC.yaml`, `experiment_config.yaml`
- `data/generation_scripts/` — input generation, reference generation, correctness checker
- `torch/correctness.py`, `torch/benchmark.py`
- `ascendc/src/matmul_kernel.asc` — Cube MatMul using `MatmulImpl`
- `ascendc/src/matmul_host.asc` — host launcher with ACL runtime
- `ascendc/CMakeLists.txt`
- `pypto/matmul_impl.py`, `pypto/test_matmul.py`, `pypto/matmul_golden.py`
- `benchmark/run_all.sh`
- `reports/` — correctness, diagnostic, final (to be populated)

## Cube Template v2
This operator is generated from `templates/cube_operator_template/`, the Cube Operator Template v2.
