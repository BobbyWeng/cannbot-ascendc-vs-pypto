# PyPTO Operator Dashboard

Unified visualization layer for operator development status, correctness, and performance.

## Quick Install (One Command)

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/BobbyWeng/cannbot-ascendc-vs-pypto/main/dashboard/download_dashboard.sh)" _ https://github.com/BobbyWeng/cannbot-ascendc-vs-pypto.git
cd cannbot-dashboard && python3 dashboard/dashboard.py --release
open dashboard/index.html
```

This clones only `dashboard/` and `reports/release/` (~2MB) via git sparse checkout. The `dashboard/dashboard.json` is pre-built with full structured data (profiler, correctness, batch scaling). The generated `index.html` embeds all data, so it works via `file://` in your browser — no HTTP server needed.

## Quick Install (via HTTP server — alternative)

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/BobbyWeng/cannbot-ascendc-vs-pypto/main/dashboard/download_dashboard.sh)" _ https://github.com/BobbyWeng/cannbot-ascendc-vs-pypto.git
cd cannbot-dashboard && python3 dashboard/dashboard.py --release
python3 -m http.server 8765 --directory .
open http://127.0.0.1:8765/dashboard/index.html
```

## Development Mode

```bash
python3 dashboard/dashboard.py
```

Scans `operators/*/` for live data. Requires the full repository checkout.

## Release Mode

```bash
python3 dashboard/dashboard.py --release
```

Reads `dashboard/dashboard.json` (pre-built by `tools/rebuild_final_data.py`) and generates `dashboard/index.html` with embedded data. Works in sparse checkout.

## Output

```
dashboard/
├── index.html         ← Open in browser (file:// or http://)
├── dashboard.json     ← Machine-readable structured data (pre-built)
├── dashboard.py       ← Generator (single entry point)
├── download_dashboard.sh  ← Sparse checkout helper
└── README.md
```

## Features

- **Overview**: Summary cards, completion progress bar, searchable/sortable operator table
- **Operator Detail**: Per-route correctness (PASS/PARTIAL/N/A), profiler B1/B32 latency, known limitations
- **Dark theme**: GitHub Actions inspired
- **No server required**: Data embedded directly in HTML — works with `file://`

## Architecture

```
tools/rebuild_final_data.py
    → reads operators/*/reports/parsed/*.json (msprof data)
    → generates dashboard/dashboard.json (with profiler + correctness + batch_scaling)

dashboard/dashboard.py --release
    → reads dashboard/dashboard.json
    → generates dashboard/index.html (embeds all data inline)
```

`dashboard.json` is the machine-readable single source of truth. `index.html` is the human-readable view.
