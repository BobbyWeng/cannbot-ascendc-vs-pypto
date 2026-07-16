# Cleanup Plan

## DELETE — Outdated v2 archives (superseded by v4)
| File | Size | Reason |
|------|------|--------|
| `cannbot_ascendc_vs_pypto_equal_v2.tar.gz` | 24 KB | Superseded by v4 |
| `cannbot_ascendc_vs_pypto_not_v2.tar.gz` | 16 KB | Superseded by v4 |
| `cannbot_ascendc_vs_pypto_or_v2.tar.gz` | 20 KB | Superseded by v4 |
| `cannbot_ascendc_vs_pypto_where_v2.tar.gz` | 16 KB | Superseded by v4 |

## DELETE — Empty directories
| Path | Reason |
|------|--------|
| `operators/relu/ascendc/build/` | Empty (no binary) |
| `operators/relu/pypto/src/output/` | Empty |
| `operators/div/output/` | Empty |
| `operators/relu/ascendc/scripts/` | Empty |
| `operators/relu/pypto/scripts/` | Empty |
| `operators/mul/pypto/scripts/` | Empty |

## DELETE — Stale files
| Path | Size | Reason |
|------|------|--------|
| `reports/runtime/npu.lock` | ~0.3 KB | Stale lock from Jul 16 07:35 |

## KEEP — With notes
- `cannbot_ascendc_vs_pypto_div_v2.tar.gz` (388 KB) — current archive for Div
- `cannbot_ascendc_vs_pypto_div_v2.tar.gz.sha256` — matching SHA256
- All v4 archives in `archives/` — CURRENT, properly slim

## Total Estimated Cleanup
- ~76 KB from outdated v2 archives
- ~5 KB from empty directories
- ~0.3 KB from stale lock file
