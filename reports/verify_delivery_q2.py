"""Check the Q2 report package against its measured evidence and registered code."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiments/exp003"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    record = json.loads((ROOT / "reports/registry/exp003.json").read_text())
    draft = json.loads((REPORT / "record.draft.json").read_text())
    assert record == draft
    assert record["scope"] == ["2"]
    checks = json.loads((REPORT / "report_build_checks.json").read_text())
    assert checks["evaluation_complete"] and checks["sections"] == 8
    for relative, expected in checks["source_hashes"].items():
        assert sha(REPORT / relative) == expected, relative
    audit = json.loads((REPORT / "evidence/full_year_audit.json").read_text())
    assert audit["status"] == "passed"
    workbook = json.loads((REPORT / "evidence/verification.json").read_text())["result2.xlsx"]
    assert workbook["sha256"] == sha(REPORT / "result2.xlsx")
    assert workbook["saved_workbook_readback"] == "passed"
    assert "matched 0 entries" in (REPORT / "workbook-previews/formula-error-scan.json").read_text()
    missing = []
    for name in ("report.md", "methods.md", "appendix.md", "decision-log.md"):
        for target in re.findall(r"\]\(([^)]+)\)", (REPORT / name).read_text()):
            if ("://" not in target and not target.startswith("#")
                    and not (REPORT / target.split("#")[0]).exists()):
                missing.append([name, target])
    assert not missing, missing
    assert (REPORT / "report.html").stat().st_size > 100000
    assert json.loads((REPORT / "app/src/data.json").read_text())["buildStatus"] == "complete"
    source = json.loads((ROOT / "data/results/exp003/evaluation_sources.json").read_text())
    # The model implementation must actually exist in the registered Git commit.
    for name in ("data.py", "train.py", "predict.py", "evaluate.py", "protocol.json"):
        relative = "experiments/problem2/exp003/" + name
        committed = subprocess.check_output(["git", "show", record["code_commit"] + ":" + relative], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == sha(ROOT / relative), relative
    result = {"status": "passed", "registered_code_commit": record["code_commit"],
              "scope": ["2"], "source_files_checked": len(checks["source_hashes"]),
              "figures": len(checks["figures"]), "missing_local_links": missing,
              "workbook_sha256": workbook["sha256"], "report_html_sha256": sha(REPORT / "report.html"),
              "evaluation_signature": source["signature"]}
    (REPORT / "delivery_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
