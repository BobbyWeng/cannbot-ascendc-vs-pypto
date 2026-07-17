#!/usr/bin/env python3
"""生成 Cannbot 中文算子 Dashboard 数据。

发布模式以 reports/release/current_release.json 为状态与正确性唯一来源；
性能数据优先使用 release 中的结构化 route profiler，缺失时才读取带
SHA256 来源状态的 operators/*/reports/parsed 文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
OPERATORS = BASE / "operators"
OUT = BASE / "dashboard"
ROUTES = ("torch", "ascendc", "pypto")
ROUTE_NAMES = {"torch": "Torch", "ascendc": "Ascend C", "pypto": "PyPTO"}
STATUS_ZH = {
    "COMPLETE": "完全完成",
    "COMPLETE_WITH_LIMITATION": "有限制完成",
    "PARTIAL": "部分完成",
    "INCOMPLETE": "未完成",
    "BLOCKED": "阻塞",
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entries(operator_dir: Path) -> dict[str, str]:
    manifest = operator_dir / "SHA256SUMS"
    if not manifest.is_file():
        return {}
    result = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            result[parts[1].lstrip("*")] = parts[0]
    return result


def result_from_ratio(passed: int, total: int, detail: str = "") -> dict:
    status = "PASS" if passed == total else "PARTIAL"
    return {
        "status": status,
        "detail": detail or f"{passed}/{total}",
        "passed": passed,
        "total": total,
    }


def parse_correctness(op: dict) -> dict:
    """将 release correctness 显式转换为三路线状态，不猜测隐含结果。"""
    result = {
        route: {"status": "N/A", "detail": "发布源未提供"}
        for route in ROUTES
    }
    routes = op.get("routes") or {}
    for route in ROUTES:
        detail = (routes.get(route) or {}).get("correctness")
        if detail:
            upper = str(detail).upper()
            status = "PASS" if "PASS" in upper else ("FAIL" if "FAIL" in upper else "N/A")
            result[route] = {"status": status, "detail": str(detail)}

    coverage = str(op.get("correctness_coverage") or "").strip()
    if not coverage:
        return result

    if re.search(r"torch\s*[/+]\s*ascendc\s*[/+]\s*pypto.*(?:full|pass)", coverage, re.I):
        for route in ROUTES:
            if result[route]["status"] == "N/A":
                result[route] = {"status": "PASS", "detail": coverage}

    if re.search(r"\bfull\s*\(all\s+3\s+routes\)", coverage, re.I):
        for route in ROUTES:
            if result[route]["status"] == "N/A":
                result[route] = {"status": "PASS", "detail": coverage}

    # ReduceSum 同时保留 FP16 与 FP32 两条 Ascend C 精度事实。
    fp16 = re.search(r"AscendC\s+FP16\s+(\d+)\s*/\s*(\d+)", coverage, re.I)
    fp32 = re.search(r"AscendC\s+FP32\s+(\d+)\s*/\s*(\d+)", coverage, re.I)
    if fp16 or fp32:
        details = []
        if fp32:
            p32, t32 = map(int, fp32.groups())
            details.append(f"FP32 {p32}/{t32} {'PASS' if p32 == t32 else 'PARTIAL'}")
            result["ascendc"] = result_from_ratio(p32, t32, "; ".join(details))
        if fp16:
            p16, t16 = map(int, fp16.groups())
            details.append(f"FP16 {p16}/{t16} {'PASS' if p16 == t16 else 'PARTIAL'}")
            if fp32:
                result["ascendc"]["detail"] = "; ".join(details)
            else:
                result["ascendc"] = result_from_ratio(p16, t16, details[0])

    labels = {"torch": "Torch", "ascendc": "AscendC", "pypto": "PyPTO"}
    for route, label in labels.items():
        if route == "ascendc" and (fp16 or fp32):
            continue
        match = re.search(rf"\b{label}\b\s*([^;]*)(?:;|$)", coverage, re.I)
        if not match or result[route]["status"] != "N/A":
            continue
        claim = match.group(1).strip(" :—-")
        ratio = re.search(
            r"(\d+)\s*/\s*(\d+)(?=\s*(?:cases?\b|bitwise\b|pass\b|$))",
            claim,
            re.I,
        )
        if ratio:
            passed, total = map(int, ratio.groups())
            result[route] = result_from_ratio(passed, total, claim)
        elif re.search(r"\b(pass|full)\b", claim, re.I):
            result[route] = {"status": "PASS", "detail": claim or coverage}
        elif re.search(r"\bfail\b", claim, re.I):
            result[route] = {"status": "FAIL", "detail": claim}

    return result


def parsed_candidate(operator: str, route: str, batch: int, manifests: dict[str, str]):
    operator_dir = OPERATORS / operator
    candidates = []
    mismatches = 0
    for suffix in ("", "FIX"):
        relative = f"reports/parsed/{route}_b{batch}{suffix}.json"
        path = operator_dir / relative
        if not path.is_file():
            continue
        actual = sha256(path)
        expected = manifests.get(relative)
        if expected and actual != expected:
            mismatches += 1
            continue
        raw = read_json(path)
        if not isinstance(raw, dict):
            continue
        primary = number(raw.get("primary_compute_kernel_us"))
        kernel_count = number(raw.get("kernel_count")) or 0
        usable = primary is not None and primary > 0 and kernel_count > 0
        candidates.append((usable, relative, actual, expected, raw))
    if not candidates:
        return None, mismatches

    usable, relative, actual, expected, raw = next(
        (item for item in candidates if item[0]), candidates[0]
    )
    integrity = "VERIFIED" if expected else "UNMANIFESTED"
    record = {
        "metric_status": "AVAILABLE" if usable else "NO_KERNEL_EVIDENCE",
        "primary_us": number(raw.get("primary_compute_kernel_us")) if usable else None,
        "all_device_us": number(raw.get("all_device_kernels_us_per_call")) if usable else None,
        "executor_us": number(raw.get("aicpu_executor_us_per_call")) if usable else None,
        "tflops": None,
        "kernel_type": raw.get("primary_kernel_type") or "N/A",
        "kernel_name": raw.get("primary_kernel_name") or "N/A",
        "kernels_per_call": number(raw.get("kernels_per_logical_call")) if usable else None,
        "block_dim": None,
        "method": raw.get("profiler_type") or "msprof",
        "source_kind": "PARSED_PROFILER",
        "source": f"operators/{operator}/{relative}",
        "sha256": actual,
        "integrity": integrity,
        "comparable": bool(usable and integrity == "VERIFIED"),
    }
    return record, mismatches


def structured_profiler_records(operator: str, route: str, route_data: dict, release_hash: str):
    profiler = route_data.get("profiler") or {}
    records = {}
    method = profiler.get("method") or "N/A"
    for key, value in profiler.items():
        match = re.fullmatch(r"b(\d+)_(?:primary_)?us", key, re.I)
        if not match or number(value) is None:
            continue
        batch = match.group(1)
        # bN_primary_us 比 bN_us 更明确；若两者并存则优先 primary。
        if batch in records and "primary" not in key.lower():
            continue
        records[batch] = {
            "metric_status": "AVAILABLE",
            "primary_us": number(value),
            "all_device_us": number(profiler.get(f"b{batch}_all_devices_us")),
            "executor_us": None,
            "tflops": number(profiler.get(f"b{batch}_TFLOPS")),
            "kernel_type": profiler.get("primary_kernel_type") or profiler.get("kernel_type") or "N/A",
            "kernel_name": profiler.get("kernel_name") or "N/A",
            "kernels_per_call": profiler.get("kernels_per_call"),
            "block_dim": profiler.get(f"blockDim_b{batch}"),
            "method": method,
            "source_kind": "CURRENT_RELEASE",
            "source": f"reports/release/current_release.json#/operators/{operator}/routes/{route}/profiler",
            "sha256": release_hash,
            "integrity": "RELEASE_SOURCE",
            "comparable": "msprof" in str(method).lower(),
        }
    return records


def load_performance(operator: str, op: dict, release_hash: str):
    batches = [int(batch) for batch in op.get("batches", [])]
    manifests = manifest_entries(OPERATORS / operator)
    records = {}
    stats = {"release_source": 0, "verified": 0, "unmanifested": 0,
             "mismatch": 0, "no_kernel_evidence": 0}
    route_sources = {}

    for route in ROUTES:
        route_data = (op.get("routes") or {}).get(route) or {}
        structured = structured_profiler_records(operator, route, route_data, release_hash)
        if structured:
            route_sources[route] = "CURRENT_RELEASE"
            for batch, record in structured.items():
                records.setdefault(batch, {})[route] = record
                stats["release_source"] += 1
            continue

        route_sources[route] = "PARSED_PROFILER"
        for batch in batches:
            record, mismatches = parsed_candidate(operator, route, batch, manifests)
            stats["mismatch"] += mismatches
            if not record:
                continue
            # ReduceSum parsed Ascend C 文件属于旧 FP16 路线；FP32 新内核尚无结构化
            # profiler，因此展示但禁止与 FP32 correctness 混合排名。
            if operator == "reduce_sum" and route == "ascendc":
                record["comparable"] = False
                record["comparison_note"] = "该记录对应旧 FP16 累积路线，不能代表新版 FP32 内核"
            records.setdefault(str(batch), {})[route] = record
            if record["integrity"] == "VERIFIED":
                stats["verified"] += 1
            else:
                stats["unmanifested"] += 1
            if record["metric_status"] != "AVAILABLE":
                stats["no_kernel_evidence"] += 1

    return {
        "primary_metric": "primary_compute_kernel_us",
        "unit": "µs",
        "batches": records,
        "route_sources": route_sources,
        "evidence_stats": stats,
    }


def build_operator(name: str, op: dict, release_hash: str, known_limitations: list[dict]):
    correctness = parse_correctness(op)
    performance = load_performance(name, op, release_hash)
    limitations = [item for item in known_limitations if item.get("operator") == name]
    if op.get("limitation"):
        limitations.insert(0, {
            "operator": name,
            "route": "all",
            "severity": "INFO",
            "description": op["limitation"],
        })

    warnings = []
    unmanifested = performance["evidence_stats"]["unmanifested"]
    if unmanifested:
        warnings.append(f"{unmanifested} 条 profiler 记录未登记到 SHA256SUMS，不参与性能排名")
    if performance["evidence_stats"]["no_kernel_evidence"]:
        warnings.append("存在零 kernel 占位文件，已转换为 N/A")
    profiler_coverage = str(op.get("profiler_coverage") or "")
    for route in ROUTES:
        has_records = any(route in batch for batch in performance["batches"].values())
        if not has_records and re.search(ROUTE_NAMES[route], profiler_coverage, re.I):
            warnings.append(f"{ROUTE_NAMES[route]}：release 有覆盖声明，但没有可展示的结构化性能记录")

    status = op.get("final_status", "INCOMPLETE")
    return {
        "name": name,
        "status": status,
        "status_zh": STATUS_ZH.get(status, status),
        "formula": op.get("formula") or "N/A",
        "shape": op.get("shape") or "N/A",
        "dtype": op.get("dtype") or "N/A",
        "batches": op.get("batches") or [],
        "precision": op.get("precision") or "N/A",
        "correctness": correctness,
        "correctness_coverage": op.get("correctness_coverage") or "未提供",
        "profiler_coverage": profiler_coverage or "未提供",
        "performance": performance,
        "limitations": limitations,
        "warnings": warnings,
        "release_notes": {
            "rc3_fix": op.get("rc3_fix"),
            "post_rc3_fix": op.get("post_rc3_fix"),
            "post_rc3_change": ((op.get("routes") or {}).get("ascendc") or {}).get("post_rc3_change"),
        },
        "completeness_gates": op.get("completeness_gates") or [],
    }


def load_release(path: Path):
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"无法读取 release JSON: {path}")
    release_hash = sha256(path)
    known_limitations = raw.get("known_limitations") or []
    operators = {
        name: build_operator(name, op, release_hash, known_limitations)
        for name, op in (raw.get("operators") or {}).items()
    }

    evidence = {"release_source": 0, "verified": 0, "unmanifested": 0,
                "mismatch": 0, "no_kernel_evidence": 0}
    for operator in operators.values():
        for key, value in operator["performance"]["evidence_stats"].items():
            evidence[key] += value

    statuses = raw.get("status_summary") or {}
    complete = sum(
        count for status, count in statuses.items()
        if str(status).startswith("COMPLETE")
    )
    pass_routes = sum(
        1 for operator in operators.values()
        for route in ROUTES
        if operator["correctness"][route]["status"] == "PASS"
    )
    return {
        "schema_version": "3.0",
        "mode": "release",
        "language": "zh-CN",
        "release_version": raw.get("release_version") or "unknown",
        "generated_at": raw.get("generated_at") or datetime.now(timezone.utc).isoformat(),
        "release_git_commit": raw.get("git_commit") or "unknown",
        "environment": raw.get("environment") or {},
        "source": {
            "release_file": str(path.relative_to(BASE)) if path.is_relative_to(BASE) else str(path),
            "release_sha256": release_hash,
            "precedence": [
                "current_release.routes.*.profiler",
                "SHA256-tracked operators/*/reports/parsed",
                "unmanifested parsed evidence (display only; never ranked)",
            ],
            "performance_matrix_used": False,
        },
        "summary": {
            "operators": len(operators),
            "complete": complete,
            "with_limitation": statuses.get("COMPLETE_WITH_LIMITATION", 0),
            "correctness_pass_routes": pass_routes,
            "total_routes": len(operators) * len(ROUTES),
            "evidence": evidence,
        },
        "status_summary": statuses,
        "operators": operators,
        "post_rc3_fixes": raw.get("post_rc3_fixes") or {},
        "known_limitations": known_limitations,
        "validation_freeze": raw.get("validation_freeze"),
        "validation_freeze_report": raw.get("validation_freeze_report"),
    }


def main():
    parser = argparse.ArgumentParser(description="生成 Cannbot 中文 Dashboard 数据")
    parser.add_argument(
        "--release",
        default="reports/release/current_release.json",
        help="发布源 JSON（默认 reports/release/current_release.json）",
    )
    args = parser.parse_args()
    release_path = Path(args.release)
    if not release_path.is_absolute():
        release_path = BASE / release_path
    if not release_path.is_file():
        print(f"[错误] 发布源不存在: {release_path}", file=sys.stderr)
        return 1

    dashboard = load_release(release_path)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "dashboard.json"
    output.write_text(
        json.dumps(dashboard, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"[完成] 版本 {dashboard['release_version']}，{dashboard['summary']['operators']} 个算子")
    print(f"[完成] Dashboard 数据: {output}")
    print(f"[提示] 运行: python3 -m http.server 8765 --directory {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
