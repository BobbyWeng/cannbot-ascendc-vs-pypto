#!/usr/bin/env python3
"""PyPTO Operator Dashboard.

Development mode:
    python dashboard.py

    Scans operators/*/ for live data. Generates dashboard/index.html + dashboard/dashboard.json.

Release mode:
    python dashboard.py --release reports/release/current_release.json

    Reads only the release JSON — does NOT scan operators/. Generates dashboard/index.html + dashboard/dashboard.json.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OPERATORS = BASE / "operators"
OUT = BASE / "dashboard"

# ---------------------------------------------------------------------------
# Release-mode data loader
# ---------------------------------------------------------------------------

def _parse_correctness_from_coverage(text):
    """Parse correctness_coverage text into per-route status dict.

    Handles patterns:
      "Torch/AscendC/PyPTO: FULL — ..."         → all three PASS
      "Torch+AscendC+PyPTO FULL (B=..)"          → all three PASS
      "Torch PASS; AscendC PASS (42/42); ..."     → individual per route
      "Torch 62/70; AscendC FP16 21/70; ..."      → partial with detail
      "FULL (all 3 routes)"                       → all three PASS (matmul)
    """
    if not text:
        return {}
    result = {}
    text_stripped = text.strip()

    # Check for combined patterns FIRST (before individual route split)
    combined_routes = ['torch/ascendc/pypto', 'torch+ascendc+pypto']
    text_lower = text_stripped.lower()
    is_combined = any(cr in text_lower for cr in combined_routes)
    is_full_all3 = text_lower.startswith('full') and '3 routes' in text_lower

    if is_combined or is_full_all3:
        # Extract the status detail after the combined header
        rest = text_stripped
        for cr in combined_routes:
            if cr in text_lower:
                idx = text_lower.index(cr) + len(cr)
                rest = text_stripped[idx:].lstrip(':').strip().lstrip('-').strip()
                break
        if is_full_all3:
            rest = text_stripped[len('FULL'):].strip('(all 3 routes)').strip(':').strip('-').strip()

        status = 'PASS'
        if rest and not rest.startswith('FULL') and 'PASS' in rest:
            status = rest
        elif rest and rest.startswith('PASS'):
            status = rest
        for label in ['torch', 'ascendc', 'pypto']:
            result[label] = status
        return result

    # Individual route patterns: split by semicolons
    parts = [p.strip() for p in text_stripped.split(';')]
    route_keys = {'torch': 'torch', 'ascendc': 'ascendc', 'pypto': 'pypto',
                  'torch+ascendc+pypto': None, 'torch/ascendc/pypto': None}
    for part in parts:
        if not part:
            continue
        part_lower = part.lower()
        matched = False
        for key in ['torch', 'ascendc', 'pypto']:
            if part_lower.startswith(key):
                rest = part[len(key):].strip().lstrip(':').strip()
                # Determine status
                if rest:
                    result[key] = rest
                else:
                    result[key] = 'PASS'
                matched = True
                break
        if not matched and 'FULL' in part:
            for label in ['torch', 'ascendc', 'pypto']:
                if label not in result:
                    result[label] = 'PASS'
    return result


def _finite_positive(val):
    """Check if val is a finite positive number."""
    if val is None:
        return False
    try:
        v = float(val)
        return v > 0 and v != float('inf') and v != float('nan')
    except (ValueError, TypeError):
        return False


_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


def validate_record(record, op_name, route_key, correctness, manifest_lookup, release_sha256):
    """Validate a profiler record against all ranking criteria.

    fail-closed: any failure → rank_eligible=false.
    Returns {"status": str, "rank_eligible": bool, "reasons": [...], ...}
    """
    reasons = []

    sk = record.get("source_kind", "")
    sha = str(record.get("sha256", "")).strip()
    integrity = str(record.get("integrity", "")).strip()
    release_sha = str(release_sha256).strip()
    corr_val = str(correctness.get(route_key, "")).strip()
    route_variant = record.get("route_variant", "")

    # 1. Correctness
    if not corr_val.startswith("PASS"):
        reasons.append("CORRECTNESS_NOT_PASS")

    # 2-4. Source + SHA256 + integrity
    if sk == "published":
        # published: must come from current_release — sha256 must equal release_sha256
        if not _SHA256_RE.match(release_sha):
            reasons.append("INVALID_RELEASE_SHA256")
        elif sha not in (release_sha, "N/A", ""):
            reasons.append("RELEASE_HASH_MISMATCH")
        elif integrity != f"sha256:{release_sha}":
            reasons.append("INTEGRITY_MISMATCH")

    elif sk == "parsed_msprof":
        expected_sha = manifest_lookup.get(op_name, {}).get(route_key)
        if not expected_sha:
            reasons.append("UNMANIFESTED")
        elif not _SHA256_RE.match(sha):
            reasons.append("INVALID_SHA256")
        elif sha != expected_sha:
            reasons.append("HASH_MISMATCH")
        elif integrity != f"sha256:{sha}":
            reasons.append("INTEGRITY_MISMATCH")

    elif sk == "parsed_msprof_git_tracked":
        reasons.append("UNMANIFESTED")

    else:
        reasons.append("MISSING_SOURCE")

    # 5. comparable
    if not record.get("comparable"):
        reasons.append("NOT_COMPARABLE")

    # 6. method
    method_raw = record.get("method", "")
    method = method_raw.split()[0] if method_raw else ""
    if method != "msprof":
        reasons.append("METHOD_MISMATCH")

    # 7. metric
    metric = str(record.get("metric", "")).strip()
    if metric != "primary_compute_kernel_us":
        reasons.append("METRIC_MISMATCH")

    # 8. primary_us finite positive
    b1 = record.get("b1_us")
    if not _finite_positive(b1):
        reasons.append("METRIC_INVALID")

    # 9. route_variant
    if route_variant and "legacy" in route_variant:
        reasons.append("WRONG_VARIANT")

    # Build result
    manifest_sha = manifest_lookup.get(op_name, {}).get(route_key)
    status = "VERIFIED" if not reasons else reasons[0]
    return {
        "status": status,
        "rank_eligible": len(reasons) == 0,
        "reasons": reasons,
        "validated_at": None,
        "source_sha256": sha if _SHA256_RE.match(sha) else None,
        "manifest_sha256": manifest_sha if _SHA256_RE.match(manifest_sha or "") else None,
        "release_sha256": release_sha if _SHA256_RE.match(release_sha) else None,
        "correctness_status": corr_val if corr_val else "N/A",
        "metric_contract": "primary_compute_kernel_us",
        "variant_match": "legacy" not in route_variant,
    }


def rank_routes(profiler_records):
    """Given all profiler records for an operator, find the fastest.

    Only records with validation.rank_eligible == True can participate.
    Requires at least 2 eligible routes to produce a winner.
    """
    eligible = [
        r for r in profiler_records.values()
        if isinstance(r, dict) and r.get("validation", {}).get("rank_eligible")
    ]
    if len(eligible) < 2:
        return {"status": "INSUFFICIENT_VERIFIED_ROUTES", "winner": None, "speedup": None}
    fastest = min(eligible, key=lambda r: float(r.get("b1_us", float('inf'))))
    return {"status": "RANKED", "winner": fastest.get("_route_key"), "speedup": None}


def _extract_profiler_from_routes(routes, route_key, dj_ops, op_name, release_sha256, op_dir, manifest_lookup):
    """Extract profiler for a route with full provenance and validation."""

    result = {"method": "N/A", "source": "N/A", "source_kind": "N/A",
              "comparable": False, "integrity": "N/A", "sha256": "N/A",
              "metric": "N/A", "_route_key": route_key}

    r = routes.get(route_key, {}) if isinstance(routes, dict) else {}
    p = r.get("profiler", {}) if isinstance(r, dict) else {}

    # Route 1: structured data from current_release.json routes.*.profiler
    if isinstance(p, dict) and (p.get("b1_us") is not None or p.get("b1_primary_us") is not None):
        for key in ["method", "b1_us", "b2_us", "b4_us", "b8_us", "b16_us", "b32_us",
                     "b1_primary_us", "b32_primary_us",
                     "b1_TFLOPS", "b32_TFLOPS", "kernels_per_call",
                     "blockDim_b1", "blockDim_b32"]:
            if key in p:
                result[key] = p[key]
        if "b1_us" not in result and "b1_primary_us" in result:
            result["b1_us"] = result["b1_primary_us"]
        if "b32_us" not in result and "b32_primary_us" in result:
            result["b32_us"] = result["b32_primary_us"]
        result["source"] = "current_release.json"
        result["source_kind"] = "published"
        result["sha256"] = release_sha256
        result["integrity"] = f"sha256:{release_sha256}"
        result["comparable"] = True
        result["metric"] = "primary_compute_kernel_us"
        if not result.get("method"):
            result["method"] = "msprof"
        return result

    # Route 2: supplementary from dashboard.json (pre-built from parsed msprof, git-tracked)
    if dj_ops and op_name in dj_ops:
        dj_prof = dj_ops[op_name].get("profiler", {}).get(route_key, {})
        if isinstance(dj_prof, dict) and dj_prof.get("b1_us") is not None:
            for key in ["method", "b1_us", "b2_us", "b4_us", "b8_us", "b16_us",
                         "b32_us", "kernel_type", "kernel_name", "kernels_per_call"]:
                if key in dj_prof:
                    result[key] = dj_prof[key]

            result["metric"] = "primary_compute_kernel_us"
            result["source"] = f"operators/{op_name}/reports/parsed/{route_key}_b1.json"

            parsed_path = op_dir / "reports" / "parsed" / f"{route_key}_b1.json"
            if parsed_path.exists():
                f_hash = hashlib.sha256(parsed_path.read_bytes()).hexdigest()
                result["sha256"] = f_hash
                result["integrity"] = f"sha256:{f_hash}"
                result["source_kind"] = "parsed_msprof"
            else:
                result["sha256"] = "N/A (not in sparse checkout)"
                result["integrity"] = "unverified (built from parsed msprof at release time)"
                result["source_kind"] = "parsed_msprof_git_tracked"

            result["comparable"] = True
            if not result.get("method"):
                result["method"] = "msprof"

            if op_name == "reduce_sum" and route_key == "ascendc":
                result["route_variant"] = "ascendc_fp16_legacy"
                result["comparable"] = False
            result["comparison_note"] = "旧 FP16 profiler，不代表新版 FP32 内核"
            return result

    return result


def load_release(path):
    """Load release-mode data source from current_release.json.

    current_release.json is the single source of truth for:
      - release_version, generated_at, environment
      - operator final_status, formula, shape, dtype, batches, precision
      - correctness_coverage text (parsed into per-route status)
      - routes.*.profiler (structured profiler data — matmul only)
      - known_limitations, ascendc_implementation_audit

    Profiler data for non-matmul operators is supplemented from
    dashboard/dashboard.json (pre-built from parsed msprof data, checked into git).
    """
    raw = json.loads(Path(path).read_text())

    # Compute SHA256 of the release file
    release_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    # Load supplementary profiler data from dashboard.json (pre-built from parsed msprof)
    dj_path = OUT / "dashboard.json"
    dj_ops = None
    if dj_path.exists():
        try:
            dj_raw = json.loads(dj_path.read_text())
            dj_ops = dj_raw.get("operators", {}) if dj_raw.get("mode") == "release" else None
        except (json.JSONDecodeError, OSError):
            pass

    # Build manifest lookup: op_name -> {route_key: expected_sha256}
    # SHA256SUMS format: <hash>  <path> or <hash> *<path>
    # Path must match relative to operator directory
    manifest_lookup = {}
    for op_name in raw.get("operators", {}):
        manifest_lookup[op_name] = {}
        sha_path = BASE / "operators" / op_name / "SHA256SUMS"
        if sha_path.exists():
            for line in sha_path.read_text().strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Support both "hash  path" and "hash *path"
                parts = line.split(None, 1)
                if len(parts) == 2:
                    f_hash = parts[0]
                    f_path = parts[1].lstrip('*')
                    # Use exact relative path match
                    for rk in ["torch", "ascendc", "pypto"]:
                        expected = f"reports/parsed/{rk}_b1.json"
                        if f_path == expected:
                            manifest_lookup[op_name][rk] = f_hash

    # Get release metadata raw — no modification
    release_version = raw.get("release_version", "unknown")
    generated_at = raw.get("generated_at", "")
    environment = raw.get("environment", {})

    ops = {}
    for name, op in raw.get("operators", {}).items():
        routes = op.get("routes", {})

        # 1. Correctness: from coverage text (primary) + routes (fallback)
        coverage_text = op.get("correctness_coverage", "")
        correctness = _parse_correctness_from_coverage(coverage_text)
        if not correctness and isinstance(routes, dict):
            for rk in ["torch", "ascendc", "pypto"]:
                rv = routes.get(rk, {})
                if isinstance(rv, dict) and rv.get("correctness"):
                    correctness[rk] = rv["correctness"]

        # 2. Profiler: routes priority, then dashboard.json
        profiler = {}
        op_dir = BASE / "operators" / name
        for rk in ["torch", "ascendc", "pypto"]:
            p = _extract_profiler_from_routes(routes, rk, dj_ops, name, release_sha256, op_dir, manifest_lookup)
            if p.get("b1_us") is not None or p.get("source") != "N/A":
                # Add validation
                correctness_initial = {}  # Will be filled below after entry built
                p["_route_key"] = rk
                profiler[rk] = p

        # 3. Batch scaling: from routes.*.profiler (per-batch keys)
        batch_scaling = {}
        for rk in ["torch", "ascendc", "pypto"]:
            r = routes.get(rk, {}) if isinstance(routes, dict) else {}
            p = r.get("profiler", {}) if isinstance(r, dict) else {}
            if isinstance(p, dict):
                bs = {}
                for bk in ["b1_us", "b2_us", "b4_us", "b8_us", "b16_us", "b32_us", "b64_us"]:
                    if bk in p and p[bk] is not None:
                        bs[bk] = p[bk]
                if bs:
                    batch_scaling[rk] = bs

        # Also try dashboard.json for batch scaling
        if not batch_scaling and dj_ops and name in dj_ops:
            dj_bs = dj_ops[name].get("batch_scaling", {})
            for rk in ["torch", "ascendc", "pypto"]:
                if rk in dj_bs and dj_bs[rk]:
                    if rk not in batch_scaling:
                        batch_scaling[rk] = dj_bs[rk]

        # 4. Build operator entry
        entry = {
            "status": op.get("final_status", "UNKNOWN"),
            "formula": op.get("formula", ""),
            "shape": op.get("shape", ""),
            "dtype": op.get("dtype", ""),
            "batches": op.get("batches", []),
            "precision": op.get("precision", ""),
            "correctness": correctness,
            "profiler": profiler,
            "batch_scaling": batch_scaling,
            "report_path": op.get("report_path", ""),
            "archive": op.get("archive", "none"),
        }

        # Add special reduce_sum annotations
        if name == "reduce_sum":
            entry["correctness_notes"] = {
                "ascendc_fp16": "21/70 PARTIAL (FP16 accum, legacy)",
                "ascendc_fp32": "70/70 PASS (FP32 accum, recommended)",
                "profiler_note": "Parsed profiler data from old FP16 kernel. FP32 kernel profiler not yet collected."
            }

        ops[name] = entry

    # Post-process: add validation + ranking for every operator
    validation_summary = {
        "verified_rankable": 0,
        "verified_not_comparable": 0,
        "unmanifested": 0,
        "hash_mismatch": 0,
        "correctness_not_pass": 0,
        "variant_mismatch": 0,
        "missing_profiler": 0,
        "insufficient_routes": 0,
    }
    for op_name, entry in ops.items():
        prof = entry.get("profiler", {})
        correctness = entry.get("correctness", {})

        # Run validate_record on each profiler entry
        for rk, rec in list(prof.items()):
            val = validate_record(rec, op_name, rk, correctness, manifest_lookup, release_sha256)
            rec["validation"] = val
            # Categorize
            if val["rank_eligible"]:
                validation_summary["verified_rankable"] += 1
            else:
                for reason in val["reasons"]:
                    if reason == "UNMANIFESTED":
                        validation_summary["unmanifested"] += 1
                    elif reason == "HASH_MISMATCH":
                        validation_summary["hash_mismatch"] += 1
                    elif reason == "CORRECTNESS_NOT_PASS":
                        validation_summary["correctness_not_pass"] += 1
                    elif reason == "WRONG_VARIANT":
                        validation_summary["variant_mismatch"] += 1
                    elif reason == "NOT_COMPARABLE":
                        validation_summary["verified_not_comparable"] += 1
                    else:
                        validation_summary.setdefault(reason, 0)
                        validation_summary[reason] += 1

        # Compute ranking
        ranking = rank_routes(prof)
        entry["ranking"] = ranking
        if ranking["status"] == "INSUFFICIENT_VERIFIED_ROUTES":
            validation_summary["insufficient_routes"] += 1

        # For profiling_data: split into rankable and display-only
        entry["profiling_display"] = {
            "rankable": {rk: p for rk, p in prof.items()
                         if p.get("validation", {}).get("rank_eligible")},
            "display_only": {rk: p for rk, p in prof.items()
                             if rk not in [k for k in prof
                                           if prof[k].get("validation", {}).get("rank_eligible")]},
        }

    # Count missing profiler
    for op_name, entry in ops.items():
        for rk in ["torch", "ascendc", "pypto"]:
            if rk not in entry.get("profiler", {}):
                validation_summary["missing_profiler"] += 1

    # Compute status summary
    status_counts = {}
    for op in ops.values():
        s = op["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    # Infer profiler_coverage from profiler availability
    profiler_coverage_summary = ""
    for name, entry in ops.items():
        prof = entry.get("profiler", {})
        routes = set(prof.keys())
        if routes:
            c = f"{','.join(sorted(routes))} profiled"
            entry["_profiler_coverage"] = c

    # Normalize release_file path to repository-relative
    path_resolved = Path(path).resolve()
    try:
        release_file_rel = str(path_resolved.relative_to(BASE))
    except ValueError:
        release_file_rel = str(path_resolved)

    validation_summary["total_operators"] = len(ops)
    result = {
        "schema_version": "1.0",
        "mode": "release",
        "release_version": release_version,
        "generated_at": generated_at,
        "environment": environment,
        "operator_count": len(ops),
        "status_summary": status_counts,
        "operators": ops,
        "validation_summary": validation_summary,
        "known_limitations": raw.get("known_limitations", []),
        "ascendc_implementation_audit": raw.get("ascendc_implementation_audit", {}),
        "source": {
            "release_file": release_file_rel,
            "release_version": release_version,
            "generated_at": generated_at,
            "release_sha256": release_sha256,
            "performance_matrix_used": False,
            "data_priority": [
                "1. current_release.json routes.*.profiler (published, exact)",
                "2. operators/*/reports/parsed/*.json (msprof, SHA256-verified at build time)",
                "3. UNMANIFESTED data: display-only, not ranked"
            ],
            "provenance": "All profiler records carry source, source_kind, sha256, integrity, and comparable per entry.",
        },
    }
    return result


# ---------------------------------------------------------------------------
# Development-mode data loader (preserved from original)
# ---------------------------------------------------------------------------

def load_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_yaml(path):
    try:
        import yaml
        return yaml.safe_load(path.read_text())
    except (ImportError, FileNotFoundError, Exception):
        return None


def safe_float(v, default=None):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def get(op_name):
    d = OPERATORS / op_name
    data = {"name": op_name, "status": "unknown", "formula": "N/A",
            "shape": "N/A", "dtype": "N/A", "batches": [],
            "precision": "N/A", "broadcast": False,
            "kernel_shape": "N/A", "logical_shape": "N/A",
            "torch": {}, "ascendc": {}, "pypto": {},
            "correctness": {}, "correctness_all_pass": None,
            "orchestrator": {}, "dev_status": {},
            "profiler": {}, "history": [],
            "last_update": "N/A", "fastest": "N/A",
            "kernel_types": {}, "kernel_count": {}}

    # SPEC
    for fname in ("SPEC.yaml", "SPEC.yml", "spec.yaml", "spec.yml"):
        spec = load_yaml(d / fname)
        if spec:
            break
    if not spec:
        spec = load_json(d / "SPEC.json") or load_json(d / "spec.json")
    if spec:
        data["formula"] = spec.get("formula", data["formula"])
        data["dtype"] = spec.get("dtype", "N/A")
        data["batches"] = spec.get("batches", [])
        if "precision" in spec and isinstance(spec["precision"], dict):
            p = spec["precision"]
            parts = []
            if p.get("rtol") is not None:
                parts.append(f"rtol={p['rtol']}")
            if p.get("atol") is not None:
                parts.append(f"atol={p['atol']}")
            if p.get("require_bitwise"):
                parts.append("bitwise")
            if p.get("note"):
                parts.append(f"({p['note']})")
            data["precision"] = ", ".join(parts) if parts else data["precision"]
        inputs = spec.get("inputs", [])
        outputs = spec.get("outputs", [])
        if inputs:
            inp = inputs[0]
            data["shape"] = inp.get("shape", inp.get("logical_shape", data["shape"]))
            data["kernel_shape"] = inp.get("kernel_shape", data["shape"])
            data["logical_shape"] = inp.get("logical_shape", data["shape"])
        if outputs:
            out = outputs[0]
            if out.get("shape") and data["shape"] == "N/A":
                data["shape"] = out["shape"]
        if spec.get("broadcast_axis") is not None or "broadcast" in spec.get("category", ""):
            data["broadcast"] = True

    # experiment_config
    exp = load_yaml(d / "experiment_config.yaml")
    if not exp:
        exp = load_yaml(d / "experiment_config.yml")
    if exp:
        data["torch"]["type"] = exp.get("implementations", {}).get("torch", {}).get("type", "N/A")
        data["ascendc"]["type"] = exp.get("implementations", {}).get("ascendc", {}).get("type", "N/A")
        data["pypto"]["type"] = exp.get("implementations", {}).get("pypto", {}).get("type", "N/A")

    # orchestrator state
    for sub in ("", "pypto/"):
        orch = load_json(d / f"{sub}.orchestrator_state.json")
        if orch:
            data["orchestrator"] = orch
            data["dev_status"] = orch.get("stage_status", {})
            break

    # final comparison
    final = load_json(d / "reports" / "final" / "final_comparison.json")
    if not final:
        final = load_json(d / "reports" / "final" / "comparison_results.json")

    if final:
        data["formula"] = final.get("formula", final.get("description", data["formula"]))
        data["shape"] = final.get("shape", data["shape"])
        data["dtype"] = final.get("dtype", data["dtype"])
        data["batches"] = final.get("experiment", final).get("batches", data["batches"])

        corr = final.get("correctness", {})
        if isinstance(corr, dict):
            all_pass = corr.get("all_batches_pass")
            if all_pass is not None:
                data["correctness_all_pass"] = all_pass
            else:
                status_val = corr.get("status")
                if status_val:
                    data["correctness_all_pass"] = status_val == "PASS"
            data["correctness"]["note"] = corr.get("note", "")
        elif isinstance(corr, str):
            data["correctness_all_pass"] = "PASS" in corr.upper()
            data["correctness"]["note"] = corr
        if data["correctness_all_pass"] is None and "results" in final:
            results = final["results"]
            impl_statuses = []
            for impl_name, impl_data in results.items():
                c = impl_data.get("correctness", "")
                if isinstance(c, str) and "PASS" in c.upper():
                    impl_statuses.append(True)
                elif isinstance(c, str) and "FAIL" in c.upper():
                    impl_statuses.append(False)
            if impl_statuses:
                data["correctness_all_pass"] = all(impl_statuses)
        if data["correctness_all_pass"] is None:
            for impl_name in ("torch", "ascendc", "pypto"):
                c = final.get("correctness", {}).get(impl_name, {})
                if isinstance(c, dict):
                    st = c.get("status", "")
                    if st == "PASS":
                        data["correctness_all_pass"] = True
                    elif st == "FAIL":
                        data["correctness_all_pass"] = False

        pdata = final.get("profiler_data", {})
        summary = final.get("comparison_summary", [])
        if not pdata and "results" in final:
            results = final["results"]
            for impl in ("torch", "ascendc", "pypto"):
                if impl in results:
                    batches_data = results[impl].get("batches", {})
                    for bk, bv in batches_data.items():
                        if bk not in pdata:
                            pdata[bk] = {}
                        if impl not in pdata[bk]:
                            pdata[bk][impl] = {}
                        pdata[bk][impl]["primary_compute_kernel_us"] = bv.get("median_us")
                        pdata[bk][impl]["all_device_kernels_us"] = bv.get("median_us")

        data["profiler_data"] = pdata
        data["comparison_summary"] = summary

        for bk, bv in pdata.items():
            for impl in ("torch", "ascendc"):
                if impl in bv:
                    d_impl = bv[impl]
                    if impl not in data["kernel_types"]:
                        data["kernel_types"][impl] = set()
                    kt = d_impl.get("kernel_type", "N/A")
                    data["kernel_types"][impl].add(kt)
                    kc = d_impl.get("kernels_per_call", 0)
                    if isinstance(kc, dict):
                        kc = kc.get("total", 0)
                    data["kernel_count"][impl] = kc
            if "pypto" in bv:
                dp = bv["pypto"]
                if "pypto" not in data["kernel_types"]:
                    data["kernel_types"]["pypto"] = set()
                pk = dp.get("primary_kernel_type", dp.get("kernel_type", "N/A"))
                ek = dp.get("executor_kernel_type", "")
                data["kernel_types"]["pypto"].add(pk)
                if ek:
                    data["kernel_types"]["pypto"].add(ek)
                kc = dp.get("kernels_per_call", 0)
                if isinstance(kc, dict):
                    kc = kc.get("total", 0)
                data["kernel_count"]["pypto"] = kc

        if summary and isinstance(summary, list):
            fastest_counts = {}
            for s in summary:
                f = s.get("fastest", "")
                if f:
                    fastest_counts[f] = fastest_counts.get(f, 0) + 1
            if fastest_counts:
                data["fastest"] = max(fastest_counts, key=fastest_counts.get)

    # raw unified summary
    u = load_json(d / "reports" / "raw" / "unified_summary.json")
    if u:
        results = u.get("results", {})
        for bk, bv in results.items():
            impls = bv.get("implementations", {})
            for impl_key, iv in impls.items():
                if "kernel" in iv:
                    k = iv["kernel"]
                    label = "ascendc" if "ascendc" in impl_key else ("pypto" if "pypto" in impl_key else "torch")
                    if label not in data:
                        data[label] = {}
                    if "benchmark" not in data[label]:
                        data[label]["benchmark"] = {}
                    data[label]["benchmark"][bk] = {
                        "median_us": k.get("median_us"),
                        "mean_us": k.get("mean_us"),
                        "min_us": k.get("min_us"),
                        "std_us": k.get("std_us"),
                        "cv_percent": k.get("cv_percent"),
                        "bw_GBps": k.get("bw_GBps"),
                    }

    # correctness from unified summary
    if u and "correctness" in u:
        for c in u["correctness"]:
            label = "ascendc" if "ascendc" in c.get("implementation", "") else ("pypto" if "pypto" in c.get("implementation", "") else "torch")
            bk = str(c.get("batch", ""))
            if label not in data["correctness"]:
                data["correctness"][label] = {}
            data["correctness"][label][bk] = {
                "status": c.get("status", "N/A"),
                "bitwise_equal": c.get("bitwise_equal"),
                "max_abs_diff": c.get("max_abs_diff"),
                "bitwise_mismatch_count": c.get("bitwise_mismatch_count"),
                "signed_zero_mismatch_count": c.get("signed_zero_mismatch_count"),
                "numeric_mismatch_count": c.get("numeric_mismatch_count"),
                "nan_count": c.get("nan_count"),
                "inf_count": c.get("inf_count"),
            }

    # parsed reports
    parsed_dir = d / "reports" / "parsed"
    if parsed_dir.exists():
        parsed_files = sorted(parsed_dir.glob("*.json"))
        data["parsed"] = {}
        for pf in parsed_files:
            pdata_parsed = load_json(pf)
            if pdata_parsed:
                stem = pf.stem
                summary = {
                    "kernel_count": pdata_parsed.get("kernel", {}).get("kernel_count", 0),
                    "total_kernel_dur_us": pdata_parsed.get("kernel", {}).get("total_kernel_dur_us", 0),
                    "by_type": pdata_parsed.get("kernel", {}).get("by_type", {}),
                    "by_name": pdata_parsed.get("kernel", {}).get("by_name", {}),
                    "kernels_per_call": pdata_parsed.get("kernel", {}).get("kernels_per_call", 0),
                    "all_kernel_dur_per_call_us": pdata_parsed.get("kernel", {}).get("all_kernel_dur_per_call_us", 0),
                }
                data["parsed"][stem] = summary

    # history
    history_dir = d / "reports" / "history"
    if history_dir.exists():
        hfiles = sorted(history_dir.glob("v*/final_comparison.json")) + sorted(history_dir.glob("v*/comparison_results.json"))
        for hf in hfiles:
            hdata = load_json(hf)
            if hdata:
                version = hf.parent.name
                data["history"].append({"version": version, "data": hdata})

    # last update
    latest_mtime = 0
    for f in d.rglob("*"):
        if f.is_file() and not f.name.startswith("."):
            try:
                mt = f.stat().st_mtime
                if mt > latest_mtime:
                    latest_mtime = mt
            except OSError:
                pass
    if latest_mtime:
        data["last_update"] = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M")

    # status — prefer release JSON final_status when available
    release_json = load_json(BASE / "reports" / "release" / "current_release.json")
    if release_json:
        op_from_release = release_json.get("operators", {}).get(op_name)
        if op_from_release:
            fs = op_from_release.get("final_status", "")
            if fs == "COMPLETE":
                data["status"] = "completed"
            elif fs == "COMPLETE_WITH_LIMITATION":
                data["status"] = "completed"
            elif fs == "PARTIAL":
                data["status"] = "in_progress"
            else:
                data["status"] = "planned"
            locked = data["orchestrator"].get("lock_status", "")
            if "BLOCKED" in locked:
                data["status"] = "blocked"
            return data

    st = data["orchestrator"].get("current_stage", 0)
    if st >= 7:
        data["status"] = "completed"
    elif st >= 1:
        data["status"] = "in_progress"
    else:
        has_torch = bool(load_json(d / "torch" / "benchmark_results.json"))
        has_ascendc = (d / "ascendc" / "build").exists() and any((d / "ascendc" / "build").iterdir())
        has_pypto_golden = (d / "pypto" / "golden").exists()
        has_final_report = bool(final)
        if has_final_report:
            data["status"] = "completed" if data.get("correctness_all_pass") is True else "in_progress"
        elif has_torch and has_ascendc:
            data["status"] = "in_progress"
        else:
            data["status"] = "planned"

    locked = data["orchestrator"].get("lock_status", "")
    if "BLOCKED" in locked:
        data["status"] = "blocked"

    # dev status detail
    dev = data["dev_status"]
    sd = {"Intent": "completed" if dev.get("1") == "completed" else "pending",
          "API": "completed" if dev.get("2") == "completed" else "pending",
          "Golden": "completed" if dev.get("3") == "completed" else "pending",
          "Design": "completed" if dev.get("4") == "completed" else "pending",
          "Implementation": "completed" if dev.get("5") == "completed" else "pending",
          "Correctness": "completed" if dev.get("6") == "completed" else "pending",
          "Benchmark": "completed" if dev.get("7") == "completed" else "pending",
          "Archive": "completed"}
    for k in sd:
        if dev.get(k.lower()):
            sd[k] = dev[k.lower()]
    data["dev_status_detail"] = sd

    # Convert sets to lists for JSON
    for impl in ("torch", "ascendc", "pypto"):
        if impl in data["kernel_types"]:
            data["kernel_types"][impl] = sorted(data["kernel_types"][impl])

    return data


def build_dev():
    ops_dirs = sorted([
        d.name for d in OPERATORS.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
    ])

    operators = []
    for name in ops_dirs:
        operators.append(get(name))

    total = len(operators)
    completed = sum(1 for o in operators if o["status"] == "completed")
    torch_done = sum(1 for o in operators if "KERNEL_AIVEC" in str(o["kernel_types"].get("torch", [])))
    ascendc_done = sum(1 for o in operators if "KERNEL_AIVEC" in str(o["kernel_types"].get("ascendc", [])))
    pypto_done = sum(1 for o in operators if o["pypto"])
    pass_count = sum(1 for o in operators if o["correctness_all_pass"] is True)
    fail_count = sum(1 for o in operators if o["correctness_all_pass"] is False)
    blocked = sum(1 for o in operators if o["status"] == "blocked")

    dashboard = {
        "mode": "development",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": total,
            "completed": completed,
            "torch_done": torch_done,
            "ascendc_done": ascendc_done,
            "pypto_done": pypto_done,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "blocked": blocked,
        },
        "operators": operators,
    }
    return dashboard


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(dashboard):
    s = dashboard.get("summary", {})
    mode = dashboard.get("mode", "development")
    total = s.get("total", dashboard.get("operator_count", 0))
    completed = s.get("completed", 0)
    if mode == "release":
        completed = sum(
            count for status, count in (dashboard.get("status_summary") or {}).items()
            if str(status).startswith("COMPLETE")
        )

    # Release mode source info
    release_info = ""
    if mode == "release":
        release_info = f'<p>发布版本 v{dashboard.get("release_version", "?")} — {dashboard["generated_at"]}</p>'

    ops_json = json.dumps(dashboard, ensure_ascii=False)
    ops_json_escaped = (ops_json
        .replace('&', '\\u0026')
        .replace('<', '\\u003c')
        .replace('>', '\\u003e')
        .replace('</script>', '<\\/script>')
        .replace('\u2028', '\\u2028')
        .replace('\u2029', '\\u2029')
    )
    css = generate_css()
    js = generate_js()

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cannbot 算子对比看板{f' v{dashboard.get("release_version", "")}' if mode == 'release' else ''}</title>
<style>{css}</style>
</head>
<body>
<div id="loading" style="display:flex;align-items:center;justify-content:center;min-height:100vh;color:var(--text-muted);font-size:18px">
  正在加载算子看板…
</div>

<div id="app" style="display:none">
  <div class="header">
    <div>
      <h1>Cannbot 算子对比看板</h1>
      <div class="subtitle">{'发布模式 · 数据源：reports/release/current_release.json' if mode == 'release' else '开发模式 · 扫描 operators/*/'}</div>
    </div>
    <div class="header-right">
      <span class="badge">{completed}/{total} 已完成</span>
      <span class="update-time" id="update-time"></span>
    </div>
  </div>

  <div class="toolbar">
    <input type="text" id="search" placeholder="搜索算子…">
    <label style="color:var(--text-muted);font-size:13px">点击表头可排序</label>
  </div>

  <div class="container">
    {release_info}

    <div class="summary-cards" id="summary-cards"></div>

    <div id="validation-summary" class="summary-cards" style="margin-bottom:12px"></div>

    <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:24px">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:13px;color:var(--text-secondary)">总体完成度</span>
        <span style="font-size:13px;font-weight:600" id="progress-text"></span>
      </div>
      <div class="progress-bar">
        <div class="fill" id="progress-fill" style="background:var(--accent-green)"></div>
      </div>
    </div>

    <table id="op-table">
      <thead>
        <tr>
          <th data-sort="name">算子 <span class="sort-arrow">▲</span></th>
          <th data-sort="status">状态 <span class="sort-arrow"></span></th>
          <th>Torch</th>
          <th>Ascend C</th>
          <th>PyPTO</th>
          <th>正确性汇总</th>
          {"<th>B=1 主计算核</th>" if mode == 'release' else '<th>性能（B=1）</th>'}
          {"<th>采样工具</th>" if mode == 'release' else '<th>最后更新</th>'}
        </tr>
      </thead>
      <tbody id="op-table-body"></tbody>
    </table>

    <div class="detail-view" id="detail-view">
      <div class="detail-header">
        <h2 id="detail-title">算子详情</h2>
        <button class="detail-close" onclick="closeDetail()">关闭</button>
      </div>

      <div class="info-grid">
        <div class="info-item"><div class="label">公式</div><div class="value" id="info-formula">-</div></div>
        <div class="info-item"><div class="label">形状</div><div class="value" id="info-shape">-</div></div>
        <div class="info-item"><div class="label">数据类型</div><div class="value" id="info-dtype">-</div></div>
        <div class="info-item"><div class="label">批次</div><div class="value" id="info-batches">-</div></div>
        <div class="info-item"><div class="label">精度标准</div><div class="value" id="info-precision">-</div></div>
        <div class="info-item"><div class="label">状态</div><div class="value" id="info-status">-</div></div>
        <div class="info-item"><div class="label">限制</div><div class="value" id="info-limitation">-</div></div>
        <div class="info-item"><div class="label">归档</div><div class="value" id="info-archive">-</div></div>
      </div>

      <div class="tabs">
        <div class="tab active" data-tab="correctness" onclick="switchTab('correctness')">正确性</div>
        <div class="tab" data-tab="performance" onclick="switchTab('performance')">性能与排名</div>
        <div class="tab" data-tab="limitations" onclick="switchTab('limitations')">限制</div>
      </div>

      <div class="tab-content active" id="tab-correctness">
        <div class="section">
          <h3>正确性汇总</h3>
          <div id="corr-status" style="margin-bottom:12px"></div>
        </div>
        <div class="section">
          <h3>各路线结果</h3>
          <table>
            <thead><tr><th>路线</th><th>结果与证据</th><th>采样信息</th></tr></thead>
            <tbody id="corr-table-body"></tbody>
          </table>
        </div>
      </div>

      <div class="tab-content" id="tab-performance">
        <div class="section">
          <h3>性能采样与验证状态</h3>
          <table>
            <thead><tr><th>路线</th><th>方法与来源</th><th>主计算核延迟</th></tr></thead>
            <tbody id="perf-table-body"></tbody>
          </table>
        </div>
      </div>

      <div class="tab-content" id="tab-limitations">
        <div class="section">
          <h3>已知限制</h3>
          <div id="limitations-content"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script type="application/json" id="dashboard-data">{ops_json_escaped}</script>
<script>{js}</script>
</body>
</html>"""


def generate_css():
    return """/* PyPTO Dashboard — Dark Theme */
:root {
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-tertiary: #21262d;
  --bg-hover: #30363d;
  --border: #30363d;
  --text-primary: #e6edf3;
  --text-secondary: #8b949e;
  --text-muted: #6e7681;
  --accent-blue: #58a6ff;
  --accent-green: #3fb950;
  --accent-yellow: #d29922;
  --accent-red: #f85149;
  --accent-purple: #bc8cff;
  --accent-orange: #f0883e;
  --accent-cyan: #39d2c0;
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  min-height: 100vh;
}

a { color: var(--accent-blue); text-decoration: none; }
a:hover { text-decoration: underline; }

.header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}
.header h1 { font-size: 20px; font-weight: 600; }
.header .subtitle { color: var(--text-secondary); font-size: 13px; margin-top: 2px; }
.header-right { display: flex; align-items: center; gap: 16px; }
.header-right .update-time { color: var(--text-muted); font-size: 12px; }
.header-right .badge { background: var(--accent-blue); color: #fff; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }

.toolbar {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.toolbar input {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 12px;
  color: var(--text-primary);
  font-size: 14px;
  width: 240px;
  outline: none;
}
.toolbar input:focus { border-color: var(--accent-blue); }

.container { max-width: 1400px; margin: 0 auto; padding: 24px; }

.summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
.summary-card { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }
.summary-card .value { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
.summary-card .label { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
.summary-card.color-blue .value { color: var(--accent-blue); }
.summary-card.color-green .value { color: var(--accent-green); }
.summary-card.color-red .value { color: var(--accent-red); }
.summary-card.color-yellow .value { color: var(--accent-yellow); }
.summary-card.color-purple .value { color: var(--accent-purple); }
.summary-card.color-orange .value { color: var(--accent-orange); }

.progress-bar { background: var(--bg-tertiary); border-radius: 6px; height: 8px; overflow: hidden; margin-top: 8px; }
.progress-bar .fill { height: 100%; border-radius: 6px; transition: width 0.3s; }

table { width: 100%; border-collapse: collapse; background: var(--bg-secondary); border-radius: 8px; overflow: hidden; margin-bottom: 24px; }
th { background: var(--bg-tertiary); padding: 10px 14px; text-align: left; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-secondary); cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { color: var(--text-primary); }
th .sort-arrow { margin-left: 4px; opacity: 0.5; }
th.sorted .sort-arrow { opacity: 1; }
td { padding: 10px 14px; border-top: 1px solid var(--border); font-size: 14px; }
tr { cursor: pointer; }
tr:hover { background: var(--bg-hover); }

.status-badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; }
.status-badge.completed { background: rgba(63,185,80,0.15); color: var(--accent-green); }
.status-badge.in_progress { background: rgba(210,153,34,0.15); color: var(--accent-yellow); }
.status-badge.planned { background: rgba(110,118,129,0.15); color: var(--text-muted); }
.status-badge.blocked { background: rgba(248,81,73,0.15); color: var(--accent-red); }
.status-badge.pass { background: rgba(63,185,80,0.15); color: var(--accent-green); }
.status-badge.fail { background: rgba(248,81,73,0.15); color: var(--accent-red); }
.status-badge.unknown { background: rgba(110,118,129,0.15); color: var(--text-muted); }

.detail-view { display: none; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 24px; margin-bottom: 24px; }
.detail-view.visible { display: block; }
.detail-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.detail-header h2 { font-size: 24px; }
.detail-close { background: none; border: 1px solid var(--border); color: var(--text-secondary); padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 14px; }
.detail-close:hover { background: var(--bg-hover); color: var(--text-primary); }

.info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }
.info-item { background: var(--bg-tertiary); border-radius: 6px; padding: 12px 16px; }
.info-item .label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.info-item .value { font-size: 15px; font-weight: 500; }

.section { margin-bottom: 24px; }
.section h3 { font-size: 16px; font-weight: 600; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }

.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
.tab { padding: 8px 16px; cursor: pointer; font-size: 13px; font-weight: 500; color: var(--text-secondary); border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab:hover { color: var(--text-primary); }
.tab.active { color: var(--accent-blue); border-bottom-color: var(--accent-blue); }
.tab-content { display: none; }
.tab-content.active { display: block; }

.limitation-item { background: var(--bg-tertiary); border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; }
.limitation-item .sev-p0 { color: var(--accent-red); font-weight: 600; }
.limitation-item .sev-p1 { color: var(--accent-yellow); font-weight: 600; }
.limitation-item .sev-p2 { color: var(--accent-orange); font-weight: 600; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.summary-card, .detail-view.visible { animation: fadeIn 0.3s ease-out; }

@media (max-width: 768px) {
  .summary-cards { grid-template-columns: repeat(2, 1fr); }
  .info-grid { grid-template-columns: repeat(2, 1fr); }
  .toolbar input { width: 160px; }
}
"""


def generate_js():
    return """// PyPTO Dashboard — Interactive
let dashboardData = null;
let currentOp = null;
let currentTab = 'correctness';
let sortField = 'name';
let sortAsc = true;

const STATUS_ZH = {
  COMPLETE: '完全完成',
  COMPLETE_WITH_LIMITATION: '有限制完成',
  PARTIAL: '部分完成',
  INCOMPLETE: '未完成',
  BLOCKED: '阻塞',
  completed: '已完成',
  in_progress: '进行中',
  planned: '计划中',
  blocked: '阻塞',
};
const ROUTE_ZH = { torch: 'Torch', ascendc: 'Ascend C', pypto: 'PyPTO' };
const VALIDATION_ZH = {
  VERIFIED: '已验证，可排名',
  UNMANIFESTED: '未入清单',
  HASH_MISMATCH: '哈希不匹配',
  INVALID_SHA256: 'SHA256 无效',
  INTEGRITY_MISMATCH: '完整性不匹配',
  RELEASE_HASH_MISMATCH: '发布源哈希不匹配',
  CORRECTNESS_NOT_PASS: '正确性未通过',
  WRONG_VARIANT: '实现版本不匹配',
  NOT_COMPARABLE: '不可比较',
  METHOD_MISMATCH: '测量方法不一致',
  METRIC_MISMATCH: '性能指标不一致',
  METRIC_INVALID: '性能数值无效',
  MISSING_SOURCE: '缺少数据来源',
};
const RANKING_ZH = {
  RANKED: '已排名',
  INSUFFICIENT_VERIFIED_ROUTES: '可信路线不足，不排名',
};
const LIMITATION_ZH = {
  'Uses bitwise_or (no logical_or API). Correct for 0/1 bool.': '缺少 logical_or API，当前使用 bitwise_or；对 0/1 布尔输入结果正确。',
  'FP16 accum precision exceeds atol=0.01 for 384-element reduction (21/70). FP32 accum kernel available (70/70).': '384 元素归约时，FP16 累加精度超过 atol=0.01（21/70）；已有 FP32 累加内核（70/70）。',
  'N=32 limits Cube utilization to ~48%. Torch faster at B≥16.': 'N=32 将 Cube 利用率限制在约 48%；B≥16 时 Torch 更快。',
  'Auto-tiling FC4000 broken. 4D JIT compile timeout for large shapes.': 'FC4000 自动分块不可用，大形状 4D JIT 编译会超时。',
  'Per-row AICPU dispatch originally ~3000 us. RC-3: torch.expand().clone() 33600x improvement.': '原逐行 AICPU 调度约 3000 µs；RC-3 使用 torch.expand().clone()，提升约 33600 倍。',
  'B=64 performance 9245us vs Torch 2180us — needs optimization.': 'B=64 时为 9245 µs，Torch 为 2180 µs，仍需优化。',
  'B=64 performance 5050us vs Torch 135us — GetValue/SetValue bottleneck.': 'B=64 时为 5050 µs，Torch 为 135 µs；瓶颈为 GetValue/SetValue。',
  'B=32 performance 329us vs Torch 113us.': 'B=32 时为 329 µs，Torch 为 113 µs。',
};

function statusZh(value) { return STATUS_ZH[value] || value || '未知'; }
function routeZh(value) { return ROUTE_ZH[value] || value; }
function validationZh(value) { return VALIDATION_ZH[value] || value || '未验证'; }
function rankingZh(value) { return RANKING_ZH[value] || value || '未排名'; }
function limitationZh(value) { return LIMITATION_ZH[value] || value || '无'; }
function correctnessZh(value) {
  return String(value || 'N/A')
    .replaceAll('COMPLETE_WITH_LIMITATION', '有限制完成')
    .replaceAll('PARTIAL', '部分通过')
    .replaceAll('PASS', '通过')
    .replaceAll('FAIL', '失败')
    .replaceAll('FULL', '全覆盖')
    .replaceAll('NEW', '新版')
    .replaceAll('all', '全部')
    .replaceAll('batches', '批次')
        .replaceAll('cases', '用例')
        .replaceAll('accum', '累加')
    .replaceAll('legacy', '旧版')
    .replaceAll('recommended', '推荐')
    .replaceAll(' for ', '，适用于 ');
}

function init() {
  var embedded = document.getElementById('dashboard-data');
  if (embedded && embedded.textContent) {
    try {
      dashboardData = JSON.parse(embedded.textContent);
      renderSummary(dashboardData);
      renderTable(dashboardData);
      setupSearch();
      setupSort();
      document.getElementById('loading').style.display = 'none';
      document.getElementById('app').style.display = 'block';
      return;
    } catch (e) {
      console.warn('嵌入数据解析失败，回退到网络加载：', e);
    }
  }
  fetch('./dashboard.json')
    .then(r => r.json())
    .then(data => {
      dashboardData = data;
      renderSummary(data);
      renderTable(data);
      setupSearch();
      setupSort();
      document.getElementById('loading').style.display = 'none';
      document.getElementById('app').style.display = 'block';
    })
    .catch(err => {
      document.getElementById('loading').textContent = '无法加载 dashboard.json：' + err.message;
    });
}

function renderSummary(data) {
  if (data.mode === 'release') {
    const s = data.status_summary || {};
    const total = data.operator_count || 0;
    const cards = [
      { label: '算子总数', value: total, cls: 'color-blue' },
    ];
    for (const [status, count] of Object.entries(s)) {
      const cls = status.startsWith('COMPLETE') ? 'color-green' : status === 'PARTIAL' ? 'color-yellow' : 'color-red';
      cards.push({ label: statusZh(status), value: count, cls: cls });
    }
    const container = document.getElementById('summary-cards');
    container.innerHTML = cards.map(c => `
      <div class="summary-card ${c.cls}">
        <div class="value">${c.value}</div>
        <div class="label">${c.label}</div>
      </div>
    `).join('');

    const compl = (s['COMPLETE'] || 0) + (s['COMPLETE_WITH_LIMITATION'] || 0);
    const pct = total > 0 ? Math.round(compl / total * 100) : 0;
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-text').textContent = pct + '% (' + compl + '/' + total + ')';
    document.getElementById('update-time').textContent = '发布时间：' + data.generated_at;

    // Validation summary
    const vs = data.validation_summary || {};
    const vContainer = document.getElementById('validation-summary');
    if (vContainer && vs.total_operators) {
      const vCards = [
        { label: '已验证且可排名', value: vs.verified_rankable || 0, cls: 'color-green' },
        { label: '已验证但不可比较', value: vs.verified_not_comparable || 0, cls: 'color-yellow' },
        { label: '未入清单', value: vs.unmanifested || 0, cls: 'color-orange' },
        { label: '哈希不匹配', value: vs.hash_mismatch || 0, cls: 'color-red' },
        { label: '正确性未通过', value: vs.correctness_not_pass || 0, cls: 'color-red' },
        { label: '实现版本不匹配', value: vs.variant_mismatch || 0, cls: 'color-red' },
        { label: '缺少性能数据', value: vs.missing_profiler || 0, cls: 'color-yellow' },
        { label: '可信路线不足', value: vs.insufficient_routes || 0, cls: 'color-yellow' },
      ];
      vContainer.innerHTML = vCards.map(c =>
        '<div class="summary-card ' + c.cls + '"><div class="value">' + c.value + '</div><div class="label">' + c.label + '</div></div>'
      ).join('');
    }
    return;
  }

  const s = data.summary;
  const cards = [
    { label: '算子总数', value: s.total, cls: 'color-blue' },
    { label: '已完成', value: s.completed, cls: 'color-green' },
    { label: 'Torch 就绪', value: s.torch_done, cls: 'color-cyan' },
    { label: 'Ascend C 就绪', value: s.ascendc_done, cls: 'color-purple' },
    { label: 'PyPTO 就绪', value: s.pypto_done, cls: 'color-orange' },
    { label: '正确性通过', value: s.pass_count, cls: 'color-green' },
    { label: '正确性失败', value: s.fail_count, cls: 'color-red' },
    { label: '阻塞', value: s.blocked, cls: 'color-yellow' },
  ];
  const container = document.getElementById('summary-cards');
  container.innerHTML = cards.map(c => `
    <div class="summary-card ${c.cls}">
      <div class="value">${c.value}</div>
      <div class="label">${c.label}</div>
    </div>
  `).join('');

  const pct = s.total > 0 ? Math.round(s.completed / s.total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent = pct + '% (' + s.completed + '/' + s.total + ')';
  document.getElementById('update-time').textContent = '生成时间：' + data.generated_at;
}

function renderTable(data) {
  if (data.mode === 'release') {
    renderReleaseTable(data);
    return;
  }
  renderDevTable(data);
}

function renderReleaseTable(data) {
  const ops = Object.entries(data.operators).map(([name, op]) => ({ name: name, ...op }));
  ops.sort((a, b) => {
    const order = { 'COMPLETE': 0, 'COMPLETE_WITH_LIMITATION': 1, 'PARTIAL': 2, 'INCOMPLETE': 3 };
    let va = (order[a.status] ?? 99);
    let vb = (order[b.status] ?? 99);
    return sortAsc ? va - vb : vb - va;
  });

  const tbody = document.getElementById('op-table-body');
  tbody.innerHTML = ops.map(op => {
    const statusCls = op.status.startsWith('COMPLETE') ? 'completed' : op.status === 'PARTIAL' ? 'in_progress' : 'unknown';
    const corr = op.correctness || {};
    const torchC = corr.torch || 'N/A';
    const ascendcC = corr.ascendc || 'N/A';
    const pyptoC = corr.pypto || 'N/A';

    const allPass = [torchC, ascendcC, pyptoC].every(c => String(c).startsWith('PASS'));
    const anyFail = [torchC, ascendcC, pyptoC].some(c => String(c).startsWith('FAIL'));
    const allNa = [torchC, ascendcC, pyptoC].every(c => c === 'N/A');
    const corrStr = allPass ? '<span class="status-badge pass">通过</span>'
      : anyFail ? '<span class="status-badge fail">失败</span>'
      : allNa ? '<span class="status-badge unknown">N/A</span>'
      : '<span class="status-badge unknown">混合 / 部分通过</span>';

    // Build perf display: only rankable data shows in the main row
    const rankable = op.profiling_display?.rankable || {};
    const displayOnly = op.profiling_display?.display_only || {};
    let b1Parts = [];
    const ranking = op.ranking || {};
    const winner = ranking.winner;

    for (const rk of ['torch', 'ascendc', 'pypto']) {
      const rp = rankable[rk];
      if (rp && rp.b1_us != null) {
        const tag = rk === winner ? '★' : '';
        b1Parts.push(tag + routeZh(rk) + ' ' + Number(rp.b1_us).toFixed(1) + ' µs');
      }
    }
    // Append display-only data with "(i)" marker
    for (const rk of ['torch', 'ascendc', 'pypto']) {
      const dp = displayOnly[rk];
      if (dp && dp.b1_us != null) {
        b1Parts.push(routeZh(rk) + ' ' + Number(dp.b1_us).toFixed(1) + ' µs（仅展示）');
      }
    }
    const b1Str = b1Parts.length > 0 ? b1Parts.join(' ') : 'N/A';

    return `<tr onclick="showDetail('${op.name}')">
      <td><strong>${op.name}</strong></td>
      <td><span class="status-badge ${statusCls}">${statusZh(op.status)}</span></td>
      <td style="font-size:12px">${correctnessZh(torchC)}</td>
      <td style="font-size:12px">${correctnessZh(ascendcC)}</td>
      <td style="font-size:12px">${correctnessZh(pyptoC)}</td>
      <td>${corrStr}</td>
      <td style="font-size:12px">${b1Str}</td>
      <td style="font-size:12px">msprof</td>
    </tr>`;
  }).join('');
}

function renderDevTable(data) {
  const ops = [...(data.operators || [])];
  ops.sort((a, b) => {
    let va = String(a[sortField] || '').toLowerCase();
    let vb = String(b[sortField] || '').toLowerCase();
    if (sortField === 'status') {
      const order = { completed: 0, in_progress: 1, planned: 2, blocked: 3 };
      const da = order[a.status] || 99;
      const db = order[b.status] || 99;
      return sortAsc ? da - db : db - da;
    }
    if (sortField === 'name') {
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    return 0;
  });

  const tbody = document.getElementById('op-table-body');
  tbody.innerHTML = ops.map(op => {
    const corr = op.correctness_all_pass === true ? '<span class="status-badge pass">通过</span>' :
                 op.correctness_all_pass === false ? '<span class="status-badge fail">失败</span>' :
                 '<span class="status-badge unknown">N/A</span>';
    const torchKt = op.kernel_types?.torch?.join(', ') || 'N/A';
    const ascendcKt = op.kernel_types?.ascendc?.join(', ') || 'N/A';
    const pyptoKt = op.kernel_types?.pypto?.join(', ') || 'N/A';
    return `<tr onclick="showDevDetail('${op.name}')">
      <td><strong>${op.name}</strong></td>
      <td><span class="status-badge ${op.status}">${statusZh(op.status)}</span></td>
      <td>${torchKt}</td>
      <td>${ascendcKt}</td>
      <td>${pyptoKt}</td>
      <td>${corr}</td>
      <td style="font-size:12px">${op.last_update}</td>
    </tr>`;
  }).join('');

  document.querySelectorAll('#op-table th').forEach(th => {
    const field = th.dataset.sort;
    th.classList.toggle('sorted', field === sortField);
    const arrow = th.querySelector('.sort-arrow');
    if (arrow && field === sortField) {
      arrow.textContent = sortAsc ? ' ▲' : ' ▼';
    }
  });
}

function setupSearch() {
  document.getElementById('search').addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#op-table-body tr');
    rows.forEach(row => {
      const text = row.textContent.toLowerCase();
      row.style.display = text.includes(q) ? '' : 'none';
    });
  });
}

function setupSort() {
  document.querySelectorAll('#op-table th').forEach(th => {
    th.addEventListener('click', () => {
      const field = th.dataset.sort;
      if (!field) return;
      if (sortField === field) {
        sortAsc = !sortAsc;
      } else {
        sortField = field;
        sortAsc = true;
      }
      renderTable(dashboardData);
    });
  });
}

function showDetail(opName) {
  if (dashboardData.mode === 'release') {
    showReleaseDetail(opName);
  } else {
    showDevDetail(opName);
  }
}

function showReleaseDetail(opName) {
  const op = dashboardData.operators[opName];
  if (!op) return;
  currentOp = op;

  document.getElementById('detail-view').classList.add('visible');
  document.getElementById('detail-title').textContent = opName + ' — 算子详情';

  document.getElementById('info-formula').textContent = op.formula || 'N/A';
  document.getElementById('info-shape').textContent = op.shape || 'N/A';
  document.getElementById('info-dtype').textContent = op.dtype || 'N/A';
  document.getElementById('info-batches').textContent = (op.batches || []).join(', ') || 'N/A';
  document.getElementById('info-precision').textContent = op.precision || 'N/A';
  const statusCls = op.status.startsWith('COMPLETE') ? 'completed' : op.status === 'PARTIAL' ? 'in_progress' : 'unknown';
  document.getElementById('info-status').innerHTML = '<span class="status-badge ' + statusCls + '">' + statusZh(op.status) + '</span>';
  document.getElementById('info-limitation').textContent = limitationZh(op.limitation);
  document.getElementById('info-archive').textContent = op.archive && op.archive !== 'none' ? op.archive : '未归档';

  renderReleaseCorrectness(op);
  renderReleasePerformance(op);

  if (dashboardData.known_limitations) {
    const opLimits = dashboardData.known_limitations.filter(l => l.operator === opName);
    const container = document.getElementById('limitations-content');
    if (opLimits.length === 0) {
      container.innerHTML = '<p style="color:var(--text-muted)">该算子暂无已知限制。</p>';
    } else {
      container.innerHTML = opLimits.map(l => {
        const sevClass = 'sev-' + l.severity.toLowerCase();
        return '<div class="limitation-item"><span class="' + sevClass + '">[' + l.severity + ']</span> ' + routeZh(l.route) + '：' + limitationZh(l.description) + '</div>';
      }).join('');
    }
  }

  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="correctness"]').classList.add('active');
  document.getElementById('tab-correctness').classList.add('active');
  document.getElementById('detail-view').scrollIntoView({ behavior: 'smooth' });
}

function renderReleaseCorrectness(op) {
  const corr = op.correctness || {};
  const vals = Object.values(corr);
  const allPass = vals.length > 0 && vals.every(c => String(c).startsWith('PASS'));
  const allFail = vals.some(c => String(c).startsWith('FAIL'));
  const allNa = vals.length === 0 || vals.every(c => c === 'N/A');
  const statusHtml = allPass ? '<span class="status-badge pass" style="font-size:16px;padding:4px 16px">全部通过</span>' :
                     allFail ? '<span class="status-badge fail" style="font-size:16px;padding:4px 16px">存在失败</span>' :
                     allNa ? '<span class="status-badge unknown" style="font-size:16px;padding:4px 16px">N/A</span>' :
                     '<span class="status-badge unknown" style="font-size:16px;padding:4px 16px">混合 / 部分通过</span>';
  document.getElementById('corr-status').innerHTML = statusHtml;

  const impls = ['torch', 'ascendc', 'pypto'];
  let html = '';
  for (const impl of impls) {
    const c = corr[impl] || 'N/A';
    const prof = op.profiler?.[impl] || {};
    const method = prof.method || 'N/A';
    const b1 = prof.b1_us != null ? ' | B=1 ' + Number(prof.b1_us).toFixed(1) + ' µs' : '';
    const src = prof.source || '';
    const srcTag = src ? '<br><span style="font-size:10px;color:var(--text-muted)">' + src + '</span>' : '';
    html += '<tr><td>' + routeZh(impl) + '</td><td>' + correctnessZh(c) + srcTag + '</td><td style="font-size:12px">' + method + b1 + '</td></tr>';
  }

  // correctness_notes (reduce_sum special)
  const notes = op.correctness_notes;
  if (notes) {
    html += '<tr><td colspan="3" style="font-size:11px;color:var(--text-secondary);padding-top:12px">';
    if (notes.ascendc_fp16) html += '<div>⚠ Ascend C FP16：' + correctnessZh(notes.ascendc_fp16) + '</div>';
    if (notes.ascendc_fp32) html += '<div>✓ Ascend C FP32：' + correctnessZh(notes.ascendc_fp32) + '</div>';
    if (notes.profiler_note) html += '<div style="margin-top:4px">ℹ 该性能数据来自旧 FP16 内核；新版 FP32 内核尚未采集 profiler。</div>';
    html += '</td></tr>';
  }

  document.getElementById('corr-table-body').innerHTML = html;
}

function renderReleasePerformance(op) {
  const profiler = op.profiler || {};
  const ranking = op.ranking || {};
  const rankable = op.profiling_display?.rankable || {};
  const displayOnly = op.profiling_display?.display_only || {};
  const winner = ranking.winner;

  const impls = ['torch', 'ascendc', 'pypto'];
  let html = '';
  for (const impl of impls) {
    const p = profiler[impl] || {};
    const val = p.validation || {};
    const rankEligible = val.rank_eligible || false;
    const isWinner = impl === winner;
    const displayOnlyData = displayOnly[impl];

    let label = routeZh(impl);
    if (isWinner) label = '★ ' + label;
    if (!rankEligible && p.b1_us != null) {
      label += '（仅展示，不排名）';
    }

    const method = p.method || 'N/A';
    const b1 = p.b1_us != null ? 'B=1 ' + Number(p.b1_us).toFixed(1) + ' µs' : 'N/A';
    const b32 = p.b32_us != null ? Number(p.b32_us).toFixed(1) + ' µs' : '';
    const lat = b32 ? b1 + ' / B=32 ' + b32 : b1;
    const src = p.source ? '<br><span style="font-size:10px;color:var(--text-muted)">' + p.source + '</span>' : '';

    // Validation badge
    let valBadge = '';
    if (val.status) {
      const valCls = val.rank_eligible ? 'pass' : 'unknown';
      valBadge = ' <span class="status-badge ' + valCls + '" style="font-size:10px">' + validationZh(val.status) + '</span>';
    }

    // Comparison note (ReduceSum FP16 legacy)
    let note = '';
    if (p.comparison_note) {
      note = '<br><span style="font-size:10px;color:var(--accent-yellow)">⚠ ' + p.comparison_note + '</span>';
    }

    html += '<tr><td>' + label + valBadge + '</td><td>' + method + src + '</td><td>' + lat + note + '</td></tr>';
  }

  // Show ranking status
  if (ranking.status) {
    html += '<tr><td colspan="3" style="font-size:11px;color:var(--text-secondary);padding-top:8px">';
    html += '排名状态：<strong>' + rankingZh(ranking.status) + '</strong>';
    if (ranking.winner) html += ' | 最快路线：<strong>' + routeZh(ranking.winner) + '</strong>';
    html += '</td></tr>';
  }

  document.getElementById('perf-table-body').innerHTML = html;
}

function closeDetail() {
  document.getElementById('detail-view').classList.remove('visible');
  currentOp = null;
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="' + tab + '"]').classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}

document.addEventListener('DOMContentLoaded', init);
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyPTO Operator Dashboard Generator")
    parser.add_argument("--release", type=str, default=None,
                        help="Release mode: path to reports/release/current_release.json")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.release:
        release_path = Path(args.release)
        if not release_path.exists():
            print(f"[ERR] Release file not found: {release_path}", file=sys.stderr)
            sys.exit(1)
        dashboard = load_release(release_path)
        print(f"[OK] Loaded {release_path} ({dashboard.get('release_version', '?')}) — {dashboard['operator_count']} operators")
    else:
        print("[INFO] Development mode — scanning operators/*/")
        dashboard = build_dev()

    # Write dashboard.json (machine-readable output)
    out_json = OUT / "dashboard.json"
    out_json.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False, default=str))
    print(f"[OK] Written {out_json}")

    # Generate index.html with embedded data
    html = generate_html(dashboard)
    out_html = OUT / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] Written {out_html}")

    print(f"\nDashboard ready: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
