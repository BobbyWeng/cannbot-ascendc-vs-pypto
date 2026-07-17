# Operator Development Workflow

This document describes the standard operator development workflow for this project. It is the parent document; route-specific details are in separate files.

## General Flow

```
User Request
    │
    ▼
┌──────────────────────────────────────────────┐
│               CLASSIFY                        │
│  Determine: backend, semantic, hardware, mode │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│              PLUGIN SELECTION                 │
│  ops-direct-invoke / pypto-orchestrator /    │
│  ops-registry-invoke / catlass / inline       │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│           PREFLIGHT (G0)                      │
│  Environment check, NPU availability          │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│           SPEC (G3)                           │
│  Operator spec, shape, dtype, precision       │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│         API ANALYSIS (G4)                     │
│  Feasibility, API mapping, constraints        │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│         GOLDEN/DATA (G5)                      │
│  Reference impl, test data generation         │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│           DESIGN (G6)                         │
│  Architecture, tiling, pipeline               │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│        IMPLEMENTATION (G7)                    │
│  Kernel code, build/JIT                       │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│        CORRECTNESS (G8 — HARD GATE)           │
│  All batches PASS within tolerance            │
│  NEVER OVERRIDABLE                            │
└──────┬───────────────────────────────────────┘
       │                                │
  PASS ▼                          FAIL  ▼
┌──────────────┐          ┌──────────────────┐
│ BASELINE     │          │ PRECISION_FIX (G9)│
│ PROFILE (G10)│          │ Re-verify         │
└──────┬───────┘          └────────┬─────────┘
       │                           │
       ▼ PASS            PASS ▼    │
┌──────────────────┐               │
│ OPTIMIZATION     │               │
│ (limited rounds) │               │
└──────┬───────────┘               │
       │                           │
       ▼                           │
┌──────────────────┐               │
│ FINAL PROFILE    │               │
│ (G12)            │               │
└──────┬───────────┘               │
       │                           │
       ▼                           ▼
┌──────────────────────────────────────────────┐
│           CODE REVIEW (G13)                   │
│  Independent reviewer, REVIEW.md              │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│           RELEASE (G14)                       │
│  pre_release_gate.sh exit 0                   │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│           ARCHIVE (G15)                       │
│  tar.gz, SHA256SUMS verification              │
└──────────────────────────────────────────────┘
```

## Concurrency Rules

- **Sequential (NPU):** correctness runs, PyPTO JIT, Ascend C runtime, benchmark, msprof, candidate timing
- **Parallel (non-NPU):** document analysis, source study, API search, parser, report generation
- **Serialization tool:** `reports/runtime/npu_run_queue.json`

## Artifact Requirements

See AGENTS.md §17 for full artifact list. Key rule: every stage produces a verifiable artifact before the next stage can begin.
