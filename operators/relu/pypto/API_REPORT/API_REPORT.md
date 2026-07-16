# API Exploration Report: ReLU

## API Mapping Table
| Operation | PyPTO API | Status |
|-----------|-----------|--------|
| ReLU | pypto.op.relu(x) | Available |
| Max with zero | pypto.maximum(x, 0) | Available (alternative) |
| Comparison | pypto.greater(x, 0) | Available (alternative) |

## Feasibility
- `pypto.op.relu` is a built-in element-wise activation in PyPTO framework
- Supports FP16 input/output
- Auto-vectorized by tile compiler
- No custom tiling required for basic cases

## Constraint Summary
- Input tensor must be contiguous ND format
- Input and output dtype must match
- No alignment constraints beyond default tensor requirements

## Recommended Implementation
Use `pypto.op.relu(x)` directly with `@pypto.frontend.jit` decorator.
