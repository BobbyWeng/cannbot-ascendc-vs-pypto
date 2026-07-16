# PyPTO Operator Generation Guide for {{ operator_name }}

## Using pypto-op-orchestrator
1. Load the `pypto-op-orchestrator` agent (see AGENTS.md)
2. The orchestrator manages the 7-stage state machine:
   - Stage 1: Intent Understanding → SPEC.md
   - Stage 2: API Exploration → API_REPORT.md
   - Stage 3: Golden Generation → {op}_golden.py
   - Stage 4: Design → DESIGN.md
   - Stage 5: Implementation → {op}_impl.py, test_{op}.py, README.md
   - Stage 6: Precision Fix (if needed)
   - Stage 7: Performance Tuning (if needed)

## Structure
```
operators/{{ operator_name }}/pypto/
├── SPEC/
├── API_REPORT/
├── DESIGN/
├── golden/
├── src/
├── tests/
├── scripts/
├── artifact_manifest.json
└── .orchestrator_state.json
```
