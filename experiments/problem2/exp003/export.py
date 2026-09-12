"""Export only result2.xlsx with the frozen template author and independent readback."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from experiments.common.neural_v2.export import prepare, verify

from .data import ROOT


def runtime_script(node, run_id):
    node = Path(node).resolve()
    packages = node.parents[1] / "node_modules"
    if not (packages / "@oai/artifact-tool").exists():
        raise RuntimeError("Use the Codex bundled Node executable with artifact-tool")
    work = ROOT / ".work" / run_id
    work.mkdir(parents=True, exist_ok=True)
    link = work / "node_modules"
    if link.is_symlink() and link.resolve() != packages:
        raise RuntimeError("Workspace Node dependency link points to a different runtime")
    if not link.exists():
        link.symlink_to(packages, target_is_directory=True)
    source = ROOT / "reports/export_workbooks_v2.mjs"
    script = work / "export_workbooks_q2.mjs"
    content = source.read_text()
    # Keep the existing author exactly, adding a bounded formula-error inspection.
    needle = "  const output=await SpreadsheetFile.exportXlsx(workbook);"
    inspection = '''  const errorScan=await workbook.inspect({searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:30},maxChars:3000,summary:"Q2 formula error scan"});
  await fs.writeFile(path.join(report,"workbook-previews","formula-error-scan.json"),JSON.stringify({records:errorScan.records,ndjson:errorScan.ndjson,truncated:errorScan.truncated},null,2));
'''
    if content.count(needle) != 1:
        raise RuntimeError("Frozen workbook author changed unexpectedly")
    script.write_text(content.replace(needle, inspection + needle))
    return script


def run(stage="all", run_id="exp003", node=None):
    out = ROOT / "data/results" / run_id
    report = ROOT / "reports/experiments" / run_id
    report.mkdir(parents=True, exist_ok=True)
    if stage != "template":
        manifest = json.loads((out / "evaluation_manifest.json").read_text())
        if manifest.get("complete") is not True:
            raise RuntimeError("Complete Q2 evaluation is required before workbook export")
    if stage in ("prepare", "all"):
        prepare(run_id, scenarios=("2",))
    if stage in ("template", "write", "all"):
        if not node:
            raise ValueError("--node must point to the bundled Codex Node executable")
        script = runtime_script(node, run_id)
        mode = "inspect" if stage == "template" else "export"
        subprocess.run([node, str(script), str(ROOT), run_id, mode, "2"], check=True)
    if stage in ("verify", "write", "all"):
        verify(run_id, scenarios=("2",))
        shutil.copy2(out / "result2.xlsx", report / "result2.xlsx")
        metadata = {
            "scenario": "2", "workbook": "result2.xlsx",
            "sha256": hashlib.sha256((out / "result2.xlsx").read_bytes()).hexdigest(),
            "author_source_sha256": hashlib.sha256(
                (ROOT / "reports/export_workbooks_v2.mjs").read_bytes()).hexdigest(),
            "verification": "Frozen independent physics/fees checks plus all workbook values and totals read back",
        }
        (out / "workbook_export.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "template", "write", "verify", "all"))
    parser.add_argument("--run-id", default="exp003")
    parser.add_argument("--node")
    args = parser.parse_args()
    run(**vars(args))
