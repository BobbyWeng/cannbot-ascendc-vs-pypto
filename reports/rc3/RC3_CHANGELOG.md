# RC-3 Changelog

## [1.3-rc3] — 2026-07-18

### Performance
- expand PyPTO: 33600x improvement (16384 AICPU → 1 materialize kernel)
- reduce_sum PyPTO: FP32 accumulation, 70/70 PASS (was 21/70)
- transpose Ascend C: +2.9% via 64×64 tile (B≥4), +13-18% from RC-2

### Profiling
- equal/not/or/where: All migrated from Event to msprof
- 52 new parsed JSON files created
- All 12 operators now have msprof coverage

### Framework
- Regression test framework created (tests/regression/)
- One-command release pipeline created (scripts/release/release.py)
- Dashboard v2 with 10 new features
- 3 critical final audit issues fixed

### Infrastructure
- SHA256: All 12 operators regenerated and verified
- SKILL_TRACE: 28 files across all operators
- All 36 regression checks passing

### Known Limitations (unchanged from RC-2)
- MatMul PyPTO: auto-tiling FC4000 (COMPLETE_WITH_LIMITATION)
- Or PyPTO: bitwise_or substitute
- Reduce_sum PyPTO: FP16 output overflow >65504
