# PyPTO Expand Operator

Implementation of Expand (Y[b,i,j] = X[b,i,0], last dim 1->384, materialized) using PyPTO framework.

- Input shape: [B, 256, 1], FP16
- Output shape: [B, 256, 384], FP16
- API: expand_wrapper(x) using pypto.op.broadcast_to
- Generated via pypto-op-orchestrator flow
