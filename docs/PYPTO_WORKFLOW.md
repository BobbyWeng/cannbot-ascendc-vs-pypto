# PyPTO Workflow

## Entry
- Plugin: `pypto-op-orchestrator` (from `/mnt/workspace/.opencode/`)
- State: `operators/{op}/.orchestrator_state.json` (official format)
- Agents: `pypto-op-analyst`, `pypto-op-developer`, `pypto-op-perf-tuner`

## 7-Stage State Machine

| Stage | Name | Skill | Agent | Artifact |
|-------|------|-------|-------|----------|
| 1 | Intent Understanding | pypto-intent-understand | (direct) | SPEC.md |
| 2 | API Exploration | pypto-api-explore | (direct) | API_REPORT.md |
| 3 | Golden Generation | pypto-golden-generate | pypto-op-analyst | {op}_golden.py |
| 4 | Design | pypto-op-design | pypto-op-analyst | DESIGN.md |
| 5 | Implementation | pypto-op-develop | pypto-op-developer | {op}_impl.py, test_{op}.py |
| 6 | Precision Fix | pypto-precision-debug, pypto-precision-compare | pypto-op-developer | Corrected impl |
| 7 | Performance Tune | pypto-op-perf-tune | pypto-op-perf-tuner | Tuned kernel |

## Stage 5 Routing

```
Implementation result
    │
    ├─ [PRECISION_PASS] → Stage 7
    │
    ├─ [PRECISION_FAIL] → Stage 6
    │
    └─ exit code ≠ 0 (no marker)
        ├─ compile error → retry Stage 5 (compile fix)
        ├─ ImportError → BLOCKED_ENVIRONMENT
        ├─ AiCore error → report, evaluate
        ├─ shape mismatch → retry Stage 5 (shape fix)
        └─ other runtime → retry Stage 5 (runtime fix)
```

**Important**: The orchestrator MUST re-run the precision test independently to confirm the agent's claim, before routing.

## Backend Limitation Handling

When PyPTO fails at backend:
1. Do NOT immediately block. Run systematic diagnosis:
2. Re-explore API with minimal shape (Stage 2).
3. Check official PyPTO samples.
4. Test shape/dtype/layout boundaries.
5. Check frontend IR.
6. Identify failing pass (lowering → CompileFunction).
7. Check version matrix (CANN, PyPTO version).
8. Try ≥3 evidence-based candidates.
9. If all fail: document as BLOCKED_BACKEND with reproduction steps.

## State File

Use the official `.orchestrator_state.json` format. DO NOT create a custom equivalent.

```json
{
  "operator_name": "{op}",
  "current_stage": 5,
  "stage_status": {"1": "completed", ...},
  "stage_retry_count": {"1": 0, ...},
  "perf_iteration": {"count": 0, ...},
  "last_updated": "2026-07-17T00:00:00Z"
}
```

## Integration with Project

```
operators/{op}/
├── pypto/
│   ├── {op}_golden.py
│   ├── {op}_impl.py
│   ├── test_{op}.py
│   └── .orchestrator_state.json
├── reports/
│   ├── correctness/    ← PyPTO correctness results
│   └── parsed/         ← PyPTO profiler results
└── ...
```
