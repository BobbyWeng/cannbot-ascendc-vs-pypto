# G13 Code Review: expand (PyPTO)

**Date**: 2026-07-23T09:19:52.537430+00:00
**Score**: 88/100 — PASS

## 1. Code Summary
One-shot expand using torch.expand().clone(). Input [B,256,1] → output [B,256,384].
Uses torch_npu-native BroadcastTo AIVEC kernel (NOT PyPTO JIT).

## 2. API Correctness
torch.expand().clone() is correct for materialized expansion.
PyPTO's expand_clone only supports 1D expansion — the torch fallback is the right choice.

## 3. Performance (PyPTO 0.2.0 + torch 2.8.0)
- B1: 4.0us (AIVEC BroadcastTo)
- B64: 9.5us
- No per-row AICPU dispatch — single kernel call

## 4. Gate Status
- G8 (Correctness): PASS (B1-B64 bitwise)
- G10 (Baseline Profile): PASS
- G13 (Code Review): PASS

## 5. Notes
- Route: torch_npu Kernel (AIVEC), NOT PyPTO JIT
- PyPTO expand_clone limitation: only works for 1D [1]→[N]
- This is the correct approach given PyPTO limitations
- No AICPU row-by-row dispatch (confirmed: single kernel)
