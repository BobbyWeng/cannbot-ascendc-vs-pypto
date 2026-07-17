# PyPTO Version Matrix

## Current Stable Version (Post-RC3)

| Component | Version/Commit |
|-----------|---------------|
| PyPTO wheel | 0.2.0 (editable install) |
| PyPTO source | `d1c290f3691effe350243f49acb0b262d0ca2e39` |
| CANN | 9.0.0 |
| torch_npu | 2.8.0.post2 |
| Orchestrator | Project-level (AGENTS.md v2) |
| Tile library | Built-in via CANN 9.0.0 |

## Known Limitations with Current PyPTO
1. **MatMul auto-tiling**: Broken for all shapes (reported as FC4000 failure); workaround: manual `set_cube_tile_shapes`
2. **MatMul 4D JIT compilation**: Extremely slow for large shapes (B=32 4D compile timeout observed)
3. **KERNEL_MIX_AIC + KERNEL_AICPU**: PyPTO MatMul uses mix mode with auxiliary kernels, adding overhead vs direct Cube
4. **Add**: Works with bitwise correctness (verified 77/77 cases)
5. **ReduceSum**: PyPTO FP32 accumulation wrapper works (casts inside kernel)

## Regression Status
No new PyPTO version is available for testing. The current version (0.2.0) is the stable baseline for Post-RC3.

### Key Operators Status
| Operator | Current Status | Notes |
|----------|---------------|-------|
| MatMul | COMPLETE_WITH_LIMITATION | PyPTO auto-tiling broken; Ascend C multi-core alternative works |
| Add | COMPLETE_WITH_LIMITATION | PyPTO backend bitwise correct; 77/77 cases pass |
| ReduceSum | COMPLETE_WITH_LIMITATION | FP32 accumulation wrapper exists |
| Logical or | COMPLETE_WITH_LIMITATION | Requires PyPTO regression |
| Expand | COMPLETE_WITH_LIMITATION | AICPU dispatch overhead |
| Broadcast Div | COMPLETE_WITH_LIMITATION | Requires verification |
| Where | COMPLETE_WITH_LIMITATION | Condition handling |
| Transpose | COMPLETE_WITH_LIMITATION | Large-shape tile |

## Recommendation
Stay on current PyPTO 0.2.0 until a new version passes:
- Official samples
- Project correctness suite
- Full profiler
- Regression tests
