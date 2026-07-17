# Ascend C Workflow

## Entry
- Plugin: `ops-direct-invoke` (from `/mnt/workspace/.opencode/`)
- Agents: `ascendc-kernel-architect`, `ascendc-kernel-design-reviewer`, `ascendc-kernel-developer`, `ascendc-kernel-reviewer`
- Skills: see `config/skill_routing.yaml` and `docs/SKILL_SELECTION_MATRIX.md`

## 7-Step Workflow

```
Step 1: Environment Check
    ascendc-env-check → environment.md
    │
    ▼ ALL PASS
Step 2: Design (Architect)
    ascendc-kernel-architect → DESIGN.md + PLAN.md
    │
    ▼
Step 2.5: Design Review
    ascendc-kernel-design-reviewer → WALKTHROUGH.md
    │
    ▼
Step 3: Development
    ascendc-kernel-developer → kernel code, build
    │
    ▼
Step 4: Code Review
    ascendc-kernel-reviewer → REVIEW.md
    │
    ├─ PASS / PASS WITH NOTES → Step 6
    │
    └─ FAIL → Step 5 (max 3 repair cycles)
    │
    ▼
Step 5: Repair Loop (max 3)
    Developer fix → Reviewer re-check
    │
    ▼ PASS
Step 6: Precision + Performance Acceptance
    6a: Reviewer → precision verification
    6b: Developer → performance collection
    │
    ▼
Step 7: Completion Report
```

## Route-Specific Directives

### Vector (hardware_path=VECTOR)
- Use Vector APIs (DataCopy, Add, Sub, etc.)
- Design must include: GM↔UB pipeline, queue/buffer, mask/repeat, alignment, tail, blockDim, totalElements, effective bandwidth
- Skills: ascendc-api-best-practices, ascendc-tiling-design, ascendc-direct-invoke-template

### Cube (hardware_path=CUBE)
- Use Cube APIs (Mmad, Matmul, Fixpipe, LoadData)
- Design must include: M/N/K, layout, L1/L0A/L0B/L0C, MMAD, TFLOPS
- Skills: ascendc-blaze-best-practice, ascendc-tiling-design, ascendc-performance-best-practices

### Performance Optimization
- G8 correctness MUST pass before any optimization
- Baseline profile via ops-profiling (msprof)
- Bottleneck classification (compute vs memory bound)
- Limited candidates per iteration (≤3)
- Each candidate passes correctness gate
- Winner needs full correctness + final profile

### Build/Runtime Repair
- Load ascendc-runtime-debug
- If crash: load ascendc-crash-debug
- Fix via developer agent (not inline by main agent)
- Re-run correctness after fix
- Re-run profiler if optimization was involved

### Precision Repair
- Load ascendc-precision-debug
- Compare against precision standard (ops-precision-standard)
- Do NOT expand tolerance to force PASS
- Review via ascendc-kernel-reviewer

## Integration with Project

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
│   │   ├── {op}_ascendc
│   │   └── output/
│   └── scripts/
├── reports/
│   ├── correctness/    ← Ascend C correctness (output BIN vs reference)
│   └── parsed/         ← Ascend C profiler data
└── ...
```
