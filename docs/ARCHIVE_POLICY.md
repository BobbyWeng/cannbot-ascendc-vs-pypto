# Archive Policy

## Purpose
Define which files belong in formal operator archives for reproducibility and which are local-only.

## Include in Archive

- **Source code**: `ascendc/src/*`, `pypto/src/*`, `pypto/golden/*`, `pypto/tests/*`
- **Configs**: `SPEC.yaml`, `experiment_config.yaml`, `manifest.json`
- **Build config**: `CMakeLists.txt`, `artifact_manifest.json`
- **Correctness results**: `reports/correctness/*`, `torch/correctness_results.json`, `pypto/correctness_results.json`
- **Profiler parsed data**: `reports/parsed/*.json`
- **Final reports**: `reports/final/*.{md,json,csv}`
- **Documentation**: `README.md`, `REPRODUCE.md`
- **Integrity**: `SHA256SUMS`
- **State**: `pypto/.orchestrator_state.json`, `ascendc/artifact_manifest.json`

## Exclude from Archive

- **Build cache**: `CMakeCache.txt`, `CMakeFiles/`, `*.o`, `*.d`, `Makefile`
- **Python cache**: `__pycache__/`, `*.pyc`
- **Raw profiler**: `reports/raw/PROF_*/` — keep LOCAL_ONLY for deep debugging
- **Large profiler data**: `profiling/PROF_*/` — keep LOCAL_ONLY
- **JIT cache**: `.pypto_cache/`, `ptx_cache/`
- **Virtual environments**: `.venv/`, `venv/`
- **Object files**: `.o`, `.os`
- **Temporary outputs**: `build/output/*.bin` (can be regenerated)

## File Size Limits

- Single files > 20 MB: exclude from formal archive; document path and purpose
- Total archive size: should not exceed 5 MB for source+config+reports

## SHA256SUMS Convention

- All paths relative to operator directory root (not project root)
- Example: `ascendc/src/div_kernel.asc`, not `operators/div/ascendc/src/div_kernel.asc`
- Only stable artifacts: source, configs, final reports, parsed data
- Regenerate when any tracked file changes

## Archive Naming

Pattern: `cannbot_ascendc_vs_pypto_{op}_v{major}.tar.gz`

Versioning:
- v1 = initial release
- v2+ = updated after significant changes (new kernels, corrected reports)
- `_pre_fix` suffix = historical, should be removed

## Cleanup

- Old archives with `_pre_fix` should be deleted once superseded
- Do not keep both v1 and v2 of the same archive if v2 fully supersedes v1
- Large archives (>50 MB) should not be committed to Git — use Git LFS or external storage
