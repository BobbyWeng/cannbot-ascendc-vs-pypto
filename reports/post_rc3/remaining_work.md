# Remaining Work After Post-RC3 Hardening

## Completed
- [x] P0-1: MatMul scheduling audit (`multicore_batch_audit.md`)
- [x] P0-2: MatMul batch-unique verification (`matmul_batch_execution_evidence.md`)
- [x] P0-3: MatMul multi-core batch implementation (1 kernel/call, blockDim=20)
- [x] P0-4: MatMul full correctness (Torch + Ascend C, all batches)
- [x] P0-5: MatMul 3-route msprof (Ascend C + Torch all batches; PyPTO B=1)
- [x] P1-1: ReduceSum FP32 device-side accumulation kernel
- [x] P1-2: ReduceSum FP16 vs FP32 accuracy comparison

## Pending

### P0-6: Add Correctness
- **Status**: BLOCKED_BACKEND (PyPTO)
- Requires PyPTO backend fix investigation
- Current `SKILL_TRACE.json` status: NON_COMPLIANT, BLOCKED_BACKEND
- Needs: test entry fix, import path resolution, full batch run

### P1-2: MatMul Tiling Optimization
- Current Ascend C already achieves 44 TFLOPS at B=32
- Potential directions (limited, ≤4 candidates):
  1. matrices_per_core tuning (affects L1 reuse)
  2. SetSingleShape manual override
  3. ND/NZ format control
  4. Double buffer tuning via SetMatmulConfigParams
- B=16 and B=32 are Torch-dominated — would need specific optimization

### P2: PyPTO Version Matrix
- Track framework commit, wheel hash, CANN version
- Compare current stable vs candidate version
- Specific regressions to check:
  1. MatMul auto tiling
  2. Expand AICPU dispatch
  3. ReduceSum FP32 accumulation
  4. logical_or
  5. BOOL/mask representation
  6. Broadcast Div
  7. Where condition handling
  8. Transpose large-shape tile

### PyPTO Profiling
- Only B=1 profiled for PyPTO
- Need B=2,4,8,16,32 with warmup=200, loops=100, repeat=5
- Two-process pattern (warmup → msprof)

### Release Gate
- Run `check_gate` for matmul, reduce_sum
- Run `verify_cannbot_usage`
- Run `pre_kernel_commit_gate`
- Run `pre_release_gate`
- Update `current_release.json` and dashboard

### Cleanup
- Remove stale MatMul/Add/ReduceSum reports from reports/raw/ (old _new suffixed)
- Update SHA256SUMS
- Update operator READMEs
