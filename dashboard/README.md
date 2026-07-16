# PyPTO Operator Dashboard

Unified visualization layer for operator development status, correctness, and performance.

## Development Mode

```bash
python dashboard.py
```

Scans `operators/*/` for live data. Requires the full repository checkout.

## Release Mode

```bash
python dashboard.py --release reports/release/current_release.json
```

Reads only the release JSON — does NOT scan operator directories.
Works with a sparse checkout containing only `dashboard/` and `reports/release/`.

## Output

```
dashboard/
├── index.html         ← Open in browser
├── dashboard.json     ← Machine-readable data
├── dashboard.py       ← Generator (single entry point)
├── download_dashboard.sh  ← Sparse checkout helper
└── README.md
```

## Features

- **Overview**: Summary cards, completion progress bar, searchable/sortable operator table
- **Operator Detail**: SPEC info, per-route correctness, profiler metrics, known limitations
- **Dark theme**: GitHub Actions inspired
- **No server required**: Pure static HTML + JavaScript

## Architecture

```
dashboard.py --release reports/release/current_release.json
    → reads release JSON (single source of truth)
    → generates dashboard/dashboard.json
    → generates dashboard/index.html (with embedded CSS/JS)
```

`dashboard.json` is consumed by `index.html`. Everything displayed comes from
`reports/release/current_release.json` — no duplicated status tables.
