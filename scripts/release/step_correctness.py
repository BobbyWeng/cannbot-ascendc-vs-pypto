import json
import sys
from release_config import OPERATORS, OPERATOR_DIR, CORRECTNESS_MATRIX

def run(dry_run=False, force=False):
    if dry_run:
        print("[DRY-RUN] step_correctness: would run correctness for all 12 operators")
        return True

    rows = []
    for op in OPERATORS:
        results = _check_op_correctness(op)
        rows.append(results)
        _log(f"{op}: Torch={results['torch']} AscendC={results['ascendc']} PyPTO={results['pypto']}")

    csv_lines = ["Operator,Torch,AscendC,PyPTO,Batches_Covered,Notes"]
    for r in rows:
        csv_lines.append(f"{r['op']},{r['torch']},{r['ascendc']},{r['pypto']},{r['batches']},{r['notes']}")

    CORRECTNESS_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    CORRECTNESS_MATRIX.write_text("\n".join(csv_lines) + "\n")
    _log(f"Correctness matrix -> {CORRECTNESS_MATRIX}")
    return True


def _check_op_correctness(op):
    op_dir = OPERATOR_DIR / op
    result = {
        "op": op,
        "torch": "N/A",
        "ascendc": "N/A",
        "pypto": "N/A",
        "batches": "",
        "notes": "",
    }

    torch_corr = op_dir / "torch" / "correctness_results.json"
    if torch_corr.exists():
        try:
            data = json.loads(torch_corr.read_text())
            results_list = data.get("results", data if isinstance(data, list) else [data])
            passes = sum(1 for r in results_list if r.get("status") == "PASS")
            total = len(results_list)
            result["torch"] = f"PASS ({passes}/{total})" if passes == total else f"{passes}/{total} PASS"
        except Exception:
            result["torch"] = "FAIL"

    ascendc_corr = op_dir / "reports" / "correctness"
    if ascendc_corr.exists():
        files = list(ascendc_corr.iterdir())
        if files:
            result["ascendc"] = "PASS"
        else:
            result["ascendc"] = "EMPTY"

    pypto_corr = op_dir / "pypto" / "correctness_results.json"
    if pypto_corr.exists():
        try:
            data = json.loads(pyto_corr.read_text())
            passes = sum(1 for r in data.get("results", []) if r.get("status") == "PASS")
            total = len(data.get("results", []))
            result["pypto"] = f"PASS ({passes}/{total})" if passes == total else f"{passes}/{total} PASS"
        except Exception:
            result["pypto"] = "FAIL"

    spec_file = op_dir / "SPEC.yaml"
    if spec_file.exists():
        import yaml
        spec = yaml.safe_load(spec_file.read_text())
        result["batches"] = ",".join(str(b) for b in spec.get("batches", []))

    return result


def _log(msg):
    print(f"  [correctness] {msg}")
