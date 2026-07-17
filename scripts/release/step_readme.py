import sys
from release_config import OPERATORS, OPERATOR_DIR, RELEASE_CHANGELOG

def run(dry_run=False, force=False):
    if dry_run:
        print("[DRY-RUN] step_readme: would update operator READMEs and RELEASE_CHANGELOG")
        return True

    count = 0
    for op in OPERATORS:
        readme_path = OPERATOR_DIR / op / "README.md"
        if readme_path.exists() and (force or _needs_update(readme_path)):
            _update_readme(readme_path, op)
            count += 1
            _log(f"{op}: README.md updated")

    if count == 0:
        _log("all READMEs up to date")
    else:
        _log(f"updated {count} README(s)")
    return True


def _needs_update(readme_path):
    content = readme_path.read_text()
    return "<!-- AUTO-GENERATED" not in content


def _update_readme(readme_path, op):
    content = readme_path.read_text()
    marker = "<!-- AUTO-GENERATED: release pipeline -->"
    if marker not in content:
        from datetime import datetime
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        content += f"\n\n{marker}\nLast verified: {now}\n"
        readme_path.write_text(content)


def _log(msg):
    print(f"  [readme] {msg}")
