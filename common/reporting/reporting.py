import json
import csv
import os
from typing import Dict, List, Any


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Recursively flatten a nested dict into dot-separated keys."""
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def generate_markdown_report(data: Dict, output_path: str) -> None:
    """Generate a structured Markdown comparison report.

    The report is organised into sections based on top-level keys in *data*.
    Nested dictionaries are rendered as indented key-value lists.

    Parameters
    ----------
    data : Dict
        Report data.  Expected structure:

        .. code-block:: python

            {
                "experiment": {"operator": ..., "shape": ..., ...},
                "ascendc": {"latency_us": ..., "bandwidth_gbs": ..., ...},
                "pypto":   {"latency_us": ..., "bandwidth_gbs": ..., ...},
                "torch_npu": {"latency_us": ..., ...},
                "correctness": {"status": ..., ...},
                "summary": {"fastest": ..., "speedup": ..., ...},
            }

    output_path : str
        Path for the output ``.md`` file.

    Raises
    ------
    OSError
        If the output directory cannot be created or the file cannot be
        written.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    lines: List[str] = []
    lines.append("# Operator Comparison Report")
    lines.append("")

    section_order = [
        "experiment",
        "correctness",
        "ascendc",
        "pypto",
        "torch_npu",
        "summary",
    ]

    seen = set(section_order)
    for section in section_order:
        if section not in data:
            continue
        lines.append(f"## {section.replace('_', ' ').title()}")
        lines.append("")
        sub = data[section]
        if isinstance(sub, dict):
            for k, v in sub.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(str(sub))
        lines.append("")

    for k, v in data.items():
        if k not in seen:
            lines.append(f"## {k.replace('_', ' ').title()}")
            lines.append("")
            if isinstance(v, dict):
                for sk, sv in v.items():
                    lines.append(f"- **{sk}**: {sv}")
            else:
                lines.append(str(v))
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_json_report(data: Dict, output_path: str) -> None:
    """Write *data* as a pretty-printed JSON file.

    Parameters
    ----------
    data : Dict
        Arbitrary serialisable report data.
    output_path : str
        Path for the output ``.json`` file.

    Raises
    ------
    OSError
        If the output directory cannot be created or the file cannot be
        written.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def generate_csv_report(data: Dict, output_path: str) -> None:
    """Flatten *data* and write as a single-row CSV file.

    Nested dictionaries are flattened with dot-separated keys, so the
    resulting CSV has a single header row and one data row.

    Parameters
    ----------
    data : Dict
        Arbitrary serialisable report data.  Values that are themselves
        dicts or lists are flattened / stringified.
    output_path : str
        Path for the output ``.csv`` file.

    Raises
    ------
    OSError
        If the output directory cannot be created or the file cannot be
        written.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    flat = _flatten_dict(data)
    header = list(flat.keys())
    row = [flat.get(h, "") for h in header]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
