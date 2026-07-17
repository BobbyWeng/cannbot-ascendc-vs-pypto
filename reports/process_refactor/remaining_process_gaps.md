# Remaining Process Gaps

This document records gaps identified during the process refactor that require follow-up.

## Gap 1: TASK_STATE.json Not Created for Existing Operators

**Severity**: HIGH
**Description**: All 12 existing operators lack `TASK_STATE.json`. The new workflow requires this file for recovery and gate tracking.
**Action**: Create a migration script that generates TASK_STATE.json for each operator by inspecting existing artifacts.

## Gap 2: NPU Run Queue Not Populated

**Severity**: MEDIUM
**Description**: `reports/runtime/npu_run_queue.json` doesn't exist. NPU scheduling was ad-hoc without serialization.
**Action**: Integrate NPU queue into benchmark scripts. The queue file structure is defined but not enforced.

## Gap 3: Code Review Gate (G13) Not Enforced for Existing Operators

**Severity**: HIGH
**Description**: No existing operator has passed through code review. The REVIEW.md artifact doesn't exist for any operator.
**Action**: Run code review for existing operators before next release. This is a pre-release blocker.

## Gap 4: Missing Correctness Reports Directory

**Severity**: MEDIUM
**Description**: `operators/{op}/reports/correctness/` exists for some operators but is empty. Correctness results are in `torch/` or `pypto/` directories instead.
**Action**: Standardize correctness report location to `operators/{op}/reports/correctness/`.

## Gap 5: No Permission Deferred File

**Severity**: LOW
**Description**: `reports/runtime/permission_deferred.json` doesn't exist. This is needed when external_directory permissions are blocked.
**Action**: Create the file with an empty array as default.

## Gap 6: Event vs msprof Cross-Comparison Not Resolved

**Severity**: HIGH
**Description**: Operators like `not`, `equal`, `or`, `where` use `torch.npu.Event` (host-synchronized) instead of msprof. The new AGENTS.md explicitly forbids cross-route ranking with different measurement levels.
**Action**: Re-profile these operators with msprof or mark them permanently as NOT_COMPARABLE in the release.

## Gap 7: Cube Operator TFLOPS Not Calculated

**Severity**: MEDIUM
**Description**: The existing Cube audit doesn't calculate TFLOPS for matmul. The new AGENTS.md requires TFLOPS = MACs / kernel_us.
**Action**: Add TFLOPS calculation to the parsing pipeline for Cube operators.

## Gap 8: Plugin Initialization Not Automated

**Severity**: LOW
**Description**: The new workflow requires loading the correct plugin. While `ops-direct-invoke` and `pypto-op-orchestrator` are installed, there's no automated step to verify the correct plugin is loaded for each task.
**Action**: Add plugin verification to G2 gate check.

## Gap 9: G8 Override Prevention Not Automated

**Severity**: HIGH
**Description**: The AGENTS.md states G8 is never overridable, but there's no mechanical enforcement — only text in the document.
**Action**: Add explicit G8 override prevention to `check_gate.py` that errors if any attempt is made to bypass G8.

## Gap 10: Historical Audit Data Cannot Feed Dashboard

**Severity**: MEDIUM
**Description**: The new AGENTS.md states that historical audits are NOT inputs to dashboard. The current `reports/release/current_release.json` may contain stale data.
**Action**: Verify current_release.json is up-to-date with actual artifact state.
