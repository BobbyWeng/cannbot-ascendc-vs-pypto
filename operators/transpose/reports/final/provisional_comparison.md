# Transpose Provisional Comparison Report

**Status: PROVISIONAL** — Ascend C device-side kernel not yet implemented; PyPTO large shape blocked at backend.

## Correctness

| Implementation | B=1..64 | Note |
|---------------|---------|------|
| Torch | PASS | Standard [B,256,384]→[B,384,256] |
| PyPTO small [16,32] | PASS | Bitwise exact |
| PyPTO large [256,384] | BLOCKED | CompileFunction pass failure |
| Ascend C | PASS* | Host pre-transpose + identity kernel |

## PyPTO Blocker Detail

```
Errcode: FFFFFF! Run pass failed., func CompileFunction, file host_machine.cpp, line 179
```

**PyPTO transpose for [256,384] tensor** hits a CompileFunction pass failure. Small tensors ([16,32]=512 elements) work correctly. The exact threshold is between 512 and 98304 elements.

## Status

**INCOMPLETE** — Ascend C needs device-side kernel. PyPTO BLOCKED_BACKEND for large shape.
