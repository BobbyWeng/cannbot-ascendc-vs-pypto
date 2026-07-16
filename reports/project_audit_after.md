# Project Audit: After Cleanup

- **Audit Time**: 2026-07-15T15:15:00Z
- **Project Path**: /mnt/workspace/cannbot_ascendc_vs_pypto
- **Total Files**: 58
- **Total Size**: ~464 KB

## Structure

| Area | Status | Description |
|------|--------|-------------|
| Project root | ✅ | README.md, AGENTS.md |
| common/ | ✅ | benchmark, correctness, profiler, reporting, schemas |
| environment/ | ✅ | environment_manifest.json, preflight.sh |
| operators/relu/ | ✅ | Complete ReLU comparison directory |
| templates/operator_template/ | ✅ | Template for new operators |
| reports/ | ✅ | Project-level audit reports |
| scripts/ | ✅ | (empty, ready for project-level scripts) |

## ReLU Directory Structure

| Subdirectory | Files | Status |
|-------------|-------|--------|
| ascendc/ | 5 src files + CMakeLists + manifest | ✅ Ascend C implementation |
| pypto/ | 8 files (SPEC, API_REPORT, DESIGN, golden, impl, tests, manifest, state) | ✅ PyPTO implementation |
| torch/ | benchmark.py, correctness.py | ✅ Torch baseline |
| data/ | manifest.json, benchmark_config.json, generation_scripts/ | ✅ Config and generators |
| benchmark/ | run_all.sh, parse_profiler.py, profiler_config/ | ✅ Benchmark infrastructure |
| reports/ | final/*.{json,csv,md}, raw/unified_summary.json | ✅ Reports with data |
| root files | SPEC.yaml, experiment_config.yaml, README.md, REPRODUCE.md, SHA256SUMS | ✅ Documentation |

## Issues Found

- [LOW] ascendc/scripts/ is empty (planned for future scripts)
- [LOW] pypto/scripts/ is empty (planned for future scripts)
- [LOW] reports/correctness/ and reports/parsed/ are empty (targets for future runs)
- [LOW] scripts/ at project root is empty (planned for project-level scripts)

## Issues Resolved vs Old Project

| Issue | Old Project | New Project |
|-------|-------------|-------------|
| aclnnRelu residue | 36 files | ❌ Removed entirely |
| Old 5D shape [B,3,4,256,32] | 2 files | ✅ Updated to [B,12,256,32] |
| manual_vmaxs as PyPTO | 28 files | ❌ Removed entirely |
| Hardcoded absolute paths | 4 files | ✅ All relative paths |
| 409MB of build cache/JIT/output | ~2329 files | ❌ Removed (regenerable) |
| Broken symlinks | 0 | ✅ None |
| Profiler raw data (26MB) | 1588 files | ❌ Not copied; can be re-generated |

## Checklist

| Item | Status |
|------|--------|
| All paths are relative | ✅ PASS |
| No hardcoded /mnt/workspace paths | ✅ PASS |
| No aclnnRelu residue | ✅ PASS |
| No manual_vmaxs residue | ✅ PASS |
| No 5D shape residue | ✅ PASS |
| No __pycache__ | ✅ PASS |
| No build cache | ✅ PASS |
| No broken symlinks | ✅ PASS |
| All scripts executable | ✅ PASS |
| SHA256SUMS present | ✅ PASS |
| REPRODUCE.md present | ✅ PASS |
| Operator template present | ✅ PASS |
