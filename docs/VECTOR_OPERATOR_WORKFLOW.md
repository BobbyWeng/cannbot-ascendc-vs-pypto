# Vector Operator Workflow

Applicable to operators classified with `hardware_path: VECTOR`.

Example operators: relu, add, mul, div, not, equal, where, reduce_sum.

## Workflow

Same as parent flow (see `OPERATOR_DEVELOPMENT_WORKFLOW.md`), with vector-specific additions:

### Preflight
- Load `ascendc-env-check` skill
- Verify NPU device available
- Verify CANN compiler available

### Design (Ascend C)
Agent: `ascendc-kernel-architect`
Skills: `ascendc-kernel-develop-workflow`, `ascendc-tiling-design`, `ascendc-api-best-practices`, `ascendc-docs-search`

Must document:
- GM↔UB pipeline design
- Queue/buffer configuration
- Vector API selection (DataCopy, Add, Sub, Mul, etc.)
- Mask/repeat strategy
- Alignment requirements
- Tail handling (remaining elements)
- blockDim computation
- TotalElements → tile count
- Batch stride
- Double buffer scheme
- Scalar fallback (if any)
- Expected effective bandwidth

### Implementation
Agent: `ascendc-kernel-developer`
Skills: `ascendc-direct-invoke-template`, `ascendc-api-best-practices`

Must use:
- Native Ascend C Vector API (no PyPTO)
- `<<<>>>` direct invoke pattern
- CMakeLists.txt from template
- Tiling header from design

### Correctness
- All formal batches must pass
- FP16: compare against FP32 torch reference
- FP32: bitwise or spec-defined tolerance
- Reduction (e.g., reduce_sum): document accumulation dtype

### Performance
- Primary metric: `primary_compute_kernel_us`
- Bandwidth: bytes / primary_compute_kernel_us
- Must report effective bandwidth

## Template Structure

```
operators/{op}/
├── ascendc/
│   ├── src/
│   │   ├── {op}_kernel.asc
│   │   ├── {op}_host.asc
│   │   ├── {op}_tiling.h
│   │   └── data_utils.h
│   ├── CMakeLists.txt
│   ├── build/
│   └── scripts/
├── torch/
│   ├── correctness.py
│   └── benchmark.py
├── pypto/          (if applicable)
├── data/
├── benchmark/
├── reports/
├── SPEC.yaml
├── TASK_STATE.json
├── SKILL_TRACE.md
├── README.md
├── REPRODUCE.md
└── SHA256SUMS
```
