# PyPTO Where Blocker Diagnostic Report

## Status: BLOCKED_BACKEND_WHERE_SELECT

## Test Results
- **[1,256,384] uint8 condition + FP16 data**: CompileFunction error: "size of tensor a (384) must match size of tensor b (3072)"
- **[1,256,384] half condition + FP16 data**: Same error — half condition also fails
- **Root cause**: PyPTO tiling Expand pass misinterprets condition physical size when dtype differs from data dtype

## Detailed Error
```
Operand shape: ([256, 384])   <- condition (384 uint8 elements or half elements)
Result shape:  ([256, 3072])  <- incorrectly expanded condition
```

384 is the condition width. 3072 = 384 × 8, which suggests the framework is expanding the condition by 8× to match some internal data size.

## Diagnosis
The PyPTO `where` operation has a tiling/shape-expansion bug when the condition tensor dtype differs in physical element size from the data tensors. Even with same-shape condition (same dtype as data), the error persists, suggesting a broader `Where` backend issue on dav-2201.

## Conclusion
Not fixable from user code. PyPTO framework does not support `pypto.where` with the required dtype compositions on dav-2201.
