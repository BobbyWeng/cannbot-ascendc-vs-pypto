# Process Refactor: Design Summary

## Overview

The process refactor replaces the original AGENTS.md (which described three routes loosely with inline skill calls) with a structured governance framework built on:

1. **Mandatory task classification** — every task is classified along 4 axes before any work.
2. **Plugin-based routing** — the correct plugin (ops-direct-invoke, pypto-op-orchestrator, etc.) is loaded as a unit, not scattered skills.
3. **15-gate lifecycle** — sequential gates with clear pass/fail criteria and NEVER-overridable correctness gate.
4. **State persistence** — every operator gets TASK_STATE.json following a shared schema.
5. **Cannbot usage verification** — automated checks ensure the framework is actually being used.
6. **Enforcement scripts** — pre-kernel-commit and pre-release gates run automatically.

## Key Design Decisions

### Why Not a Single Orchestrator?
The project compares three backends (Torch, Ascend C, PyPTO) — they have fundamentally different workflows. Forcing them into a single orchestrator would either be too abstract to be useful or too complex to maintain. Instead, each route uses its existing plugin's workflow, and the project-level framework adds governance gates around them.

### Why 15 Gates?
Each gate corresponds to a distinct lifecycle stage. Having explicit gates means:
- Clear pass/fail for each stage
- No skipping stages
- Gate G8 (correctness) is explicitly never overridable — not even via an override mechanism
- Gate G13 (code review) can be overridden but the override must be documented and results are DRAFT only

### Why TASK_STATE.json in Addition to Orchestrator State?
The `.orchestrator_state.json` is PyPTO-specific. The project-level `TASK_STATE.json` covers all backends uniformly and follows a shared JSON Schema.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    PROJECT GOVERNANCE FRAMEWORK                    │
│  task_context.json, TASK_STATE.json, gates.yaml, verify tools     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐      │
│  │ Torch Baseline │  │ Ascend C Route │  │ PyPTO Route    │      │
│  │ (inline)       │  │ ops-direct-    │  │ pypto-op-      │      │
│  │                │  │ invoke plugin   │  │ orchestrator   │      │
│  └────────────────┘  └────────────────┘  └────────────────┘      │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                GATE LAYER (G0-G15)                       │    │
│  │  Environment → Classify → Spec → API → Golden → Design   │    │
│  │  → Build → Correctness → Precision → Profile → Optimize  │    │
│  │  → Final Profile → Review → Release → Archive            │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              ENFORCEMENT LAYER                            │    │
│  │  verify_cannbot_usage.py, pre_kernel_commit_gate.sh,     │    │
│  │  pre_release_gate.sh, check_gate.py                      │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```
