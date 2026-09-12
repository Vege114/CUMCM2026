"""Three comparable Q2 template workbooks, authored with artifact-tool."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from experiments.common.neural_v2.export import prepare, verify

from .data import OUT, ROOT, VARIANTS, sha256, write_json


def run(node):
    began = time.monotonic()
    report = ROOT / "reports/experiments/exp004"
    work = ROOT / ".work/exp004"
    work.mkdir(parents=True, exist_ok=True)
    packages = Path(node).resolve().parents[1] / "node_modules"
    if not (packages / "@oai/artifact-tool").exists():
        raise RuntimeError("The Codex bundled artifact-tool runtime is required")
    link = work / "node_modules"
    if not link.exists():
        link.symlink_to(packages, target_is_directory=True)
    source = (ROOT / "reports/export_workbooks_v2.mjs").read_text()
    needle = "  const output=await SpreadsheetFile.exportXlsx(workbook);"
    scan = '''  const scan = await workbook.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:300},maxChars:2500,summary:"final formula error scan"});
  await fs.writeFile(path.join(report,"workbook-previews","formula-errors.json"),JSON.stringify(scan,null,2));
'''
    if source.count(needle) != 1:
        raise RuntimeError("Unexpected template author structure")
    script = work / "export_workbooks_exp004.mjs"
    script.write_text(source.replace(needle, scan + needle))
    checks, markdown = {}, ["# 三组预测的指定四日结果\n\n共同调度，种子42。金额为元，电量为kWh。全年季节组含未来信息。\n"]
    for variant in VARIANTS:
        target = OUT / variant
        target.mkdir(parents=True, exist_ok=True)
        (report / variant).mkdir(parents=True, exist_ok=True)
        for filename, destination in (("dispatch_2.npz", f"../dispatch_{variant}_seed_42.npz"),
                                      ("warmup_2.npz", "../../exp003/warmup_2.npz")):
            path = target / filename
            if not path.exists():
                path.symlink_to(destination)
        run_id = f"exp004/{variant}"
        prepare(run_id, ["2"])
        subprocess.run([str(node), str(script), str(ROOT), run_id, "export", "2"], check=True)
        verify(run_id, ["2"])
        shutil.copy2(target / "result2.xlsx", report / variant / "result2.xlsx")
        checks[variant] = json.loads((target / "verification.json").read_text())
        label = {"no_season": "无季节处理", "causal_season": "历史季节预判", "oracle_season": "全年季节探索（含未来信息）"}[variant]
        text = (report / variant / "specified_dates.md").read_text()
        markdown.append(f"\n## {label}\n" + text.replace("# 指定四日完整结果", "").replace("\n## 问题", "\n### 问题").replace("\n### 表", "\n#### 表"))
    (report / "specified_dates.md").write_text("\n".join(markdown))
    write_json(OUT / "workbook_export.json", {"status": "passed", "variants": checks,
               "seconds": time.monotonic() - began, "author": "@oai/artifact-tool; frozen v2 template author with error scan",
               "template_sha256": sha256(ROOT / "data/templates/result2.xlsx")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True)
    run(**vars(parser.parse_args()))
