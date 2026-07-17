# Cube Operator Workflow

Applicable to operators classified with `hardware_path: CUBE`.

Example operators: matmul, batch_matmul, linear, qk, pv, ffn, gemm.

## Workflow

Same as parent flow (see `OPERATOR_DEVELOPMENT_WORKFLOW.md`), with Cube-specific additions.

### Decision Tree

```
CUBE operator requested
    │
    ├─ Catlass explicitly requested → catlass-op-generator
    │   Agents: catlass-op-architect → catlass-op-generator → catlass-op-reviewer
    │
    ├─ PyPTO Cube → pypto-op-orchestrator
    │   Standard 7-stage PyPTO flow
    │
    └─ Ascend C native Cube → ops-direct-invoke
        Standard 7-step flow + Cube audit requirements
```

### Design (Ascend C)
Agent: `ascendc-kernel-architect`
Skills: `ascendc-kernel-develop-workflow`, `ascendc-tiling-design`, `ascendc-api-best-practices`, `ascendc-blaze-best-practice`, `ascendc-docs-search`

Must document:
- M, N, K dimensions
- Matrix count per logical call
- A, B, C layout (ND/NZ)
- L1 buffer usage
- L0A/L0B/L0C configuration
- Cube MMAD setup
- Fixpipe configuration
- Accumulation dtype
- Tile M/N/K sizes
- blockDim assignment
- Matrix per core
- Double buffer depth
- Pipeline depth
- Format conversion requirements
- Workspace requirements
- Epilogue operations

### Implementation
Agent: `ascendc-kernel-developer`
Skills: `ascendc-direct-invoke-template`, `ascendc-blaze-best-practice`

### Performance
Must report:
- TFLOPS (calculated: MACs / primary_compute_kernel_us)
- MACs: 2 × M × N × K per logical MatMul
- Format conversion cost
- Cube utilization

**Forbidden:**
- Vector elementwise mul as MatMul
- ACLNN wrapper as custom Cube
- Host MatMul
- Kernel latency without TFLOPS
- Hardcoded tile without shape boundary documentation

### Epilogue Handling
For operators with epilogue (e.g., bias add, activation after MatMul):
- Document epilogue in Cube pipeline
- Use Fixpipe for L0C→GM conversion
- Account for epilogue in performance measurement
- Do NOT measure epilogue separately from main kernel
