# Cleanup Plan

## Files to DELETE

### Outdated v2 archives (superseded by v4)
| File | Size | Reason |
|------|------|--------|
| `cannbot_ascendc_vs_pypto_equal_v2.tar.gz` | 24 KB | Superseded by v4; contains __pycache__ |
| `cannbot_ascendc_vs_pypto_not_v2.tar.gz` | 16 KB | Superseded by v4; contains __pycache__ |
| `cannbot_ascendc_vs_pypto_or_v2.tar.gz` | 20 KB | Superseded by v4; contains __pycache__ |
| `cannbot_ascendc_vs_pypto_where_v2.tar.gz` | 16 KB | Superseded by v4; contains __pycache__ |

### Empty directories
| Path | Reason |
|------|--------|
| `operators/relu/ascendc/build/` | Empty (no binary) |
| `operators/relu/pypto/src/output/` | Empty |
| `operators/div/output/` | Empty |
| `operators/relu/ascendc/scripts/` | Empty |
| `operators/relu/pypto/scripts/` | Empty |
| `operators/mul/pypto/scripts/` | Empty |
| `scripts/` (root) | Empty |

### Stale report backup
| Path | Size | Reason |
|------|------|--------|
| `operators/expand/reports/final/final_comparison.md.bak` | ~2 KB | Identical to live file |

### NPU lock file (stale)
| Path | Reason |
|------|--------|
| `reports/runtime/npu.lock` | Stale lock from Jul 16 07:35; PID 311096 no longer running |

## Files to REGENERATE (slim)

### Current archives contain build cache — need slim version
Not applicable — v4 archives are already slim.

## Files to KEEP (with caveats)

### v2 root-level archives for div
- `cannbot_ascendc_vs_pypto_div_v2.tar.gz` (388 KB) — current, needed
- No v4 div archive exists yet

### Current source archives in archives/
- All v4 archives are current and properly slim (no build cache, no __pycache__)
- SHA256 files present and correct

## Total Estimated Cleanup
- ~76 KB from outdated v2 archives
- ~5 KB from empty dirs + stale files
- ~0.5 KB from stale lock file
