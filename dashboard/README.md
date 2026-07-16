# PyPTO Operator Dashboard

Unified visualization layer for all operator development status, correctness, performance, and profiling data.

## Quick Start

```bash
python dashboard/dashboard.py
```

Open `dashboard/index.html` in any browser (no server required).

## What It Does

- Scans `operators/*/` automatically
- Reads SPEC, reports, benchmarks, profiler data, correctness results
- Generates a static HTML dashboard + JSON data file
- New operators appear automatically — no manual maintenance

## Output

```
dashboard/
├── index.html       ← Open in browser
├── dashboard.json   ← Machine-readable data
├── dashboard.py     ← Scanner/generator (single entry point)
├── dashboard.css    ← Styles (embedded in HTML, also standalone)
├── dashboard.js     ← Interactive logic (embedded in HTML, also standalone)
├── assets/          ← Reserved for future assets
└── README.md
```

## Features

- **Overview**: Summary cards, progress bar, searchable/sortable operator table
- **Operator Detail**: SPEC info, development pipeline, tabs for:
  - Correctness (per-batch table + heatmap)
  - Performance (latency per batch)
  - Side-by-side comparison (Torch vs Ascend C vs PyPTO)
  - Kernel timeline visualization
  - Kernel type distribution (pie chart)
  - Profiler data
  - Version history
- **Dark theme**: Inspired by GitHub Actions, W&B, TensorBoard
- **No server required**: Pure static HTML + JavaScript

## Architecture

```
dashboard.py  →  scans operators/*/
              →  generates dashboard.json
              →  generates index.html (with embedded CSS/JS)
```

`dashboard.json` is the single source of truth consumed by `index.html`.

## Adding a New Operator

Simply add the operator directory under `operators/` with the standard structure:

```
operators/{op}/
├── SPEC.yaml
├── experiment_config.yaml
├── torch/          (benchmark_results.json, correctness_results.json)
├── ascendc/        (src/, build/)
├── pypto/          (golden/, src/, tests/)
└── reports/
    ├── raw/
    ├── parsed/
    └── final/
        └── final_comparison.json
```

Then run `python dashboard/dashboard.py` — the new operator appears automatically.
