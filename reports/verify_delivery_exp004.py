"""Check report values, provenance, history coverage and packaged artifacts."""

import re

import numpy as np
import pandas as pd

from experiments.problem2.exp004.data import OUT, ROOT, VARIANTS, sha256, write_json
from experiments.problem2.exp004.verify import run as verify_results
from reports.exp004_evidence import EVIDENCE, REPORT, read


def run():
    verify_results()
    record = read(REPORT / "record.json")
    assert record == read(ROOT / "reports/registry/exp004.json")
    snapshot = read(REPORT / "app/src/data.json")
    build = read(REPORT / "report_build.json")
    assert snapshot["buildStatus"] == build["buildStatus"] == "complete"
    assert sha256(REPORT / "report.html") == build["html_sha256"]
    assert sha256(REPORT / "app/src/data.json") == build["data_sha256"]
    for name, expected in build["evidence"].items():
        assert sha256(EVIDENCE / name) == expected, name
    assert len(snapshot["reportContent"]) == 8
    components = [block for section in snapshot["reportContent"] for block in section["blocks"]]
    assert len({b["id"] for b in components}) == len(components)
    for query in snapshot["queries"].values():
        for name in query["source"]["files"]:
            assert (REPORT / name).is_file(), name
    for block in components:
        for qid in ([block["queryId"]] if "queryId" in block else block.get("queryIds", [])):
            assert snapshot["queries"][qid]["rows"], qid
    costs = pd.read_csv(EVIDENCE / "cost_history.csv")
    assert set(costs.experiment) == {"exp001", "exp002", "exp003", "exp004"}
    assert costs[costs.exploratory].ranking_allowed.eq(False).all()
    assert costs[~costs.physically_comparable].ranking_allowed.eq(False).all()
    current = pd.read_csv(OUT / "dispatch_metrics.csv")
    for row in costs[costs.experiment == "exp004"].itertuples():
        source = current[(current.name == row.name) & (current.seed == 42)].iloc[0]
        for key in ["planned_kwh", "emergency_kwh", "planned_cost", "emergency_cost", "total_cost"]:
            np.testing.assert_allclose(getattr(row, key), source[key], rtol=1e-12)
    comparison = pd.read_csv(EVIDENCE / "relative_comparison.csv")
    comparable = comparison[comparison.comparable]
    np.testing.assert_allclose(comparable.absolute_change, comparable.current-comparable.previous, atol=1e-7)
    nonzero = comparable[comparable.previous != 0]
    np.testing.assert_allclose(nonzero.relative_change_pct, 100*(nonzero.current-nonzero.previous)/nonzero.previous.abs(), atol=1e-8)
    assert comparison[~comparison.comparable].relative_change_pct.isna().all()
    charts = [b for b in components if b["type"] == "chart"]
    for block in charts:
        rows = snapshot["queries"][block["queryId"]]["rows"]
        fields = block["spec"].get("fields", [block["spec"]["y"]])
        assert any(all(r.get(k) is not None for k in fields) for r in rows), block["id"]
    links = 0
    for file in [REPORT / "report.md", REPORT / "README-prediction.md"]:
        for link in re.findall(r'\]\(([^)]+)\)', file.read_text()):
            if "://" not in link and not link.startswith("#"):
                assert (file.parent / link.split("#")[0]).exists(), (file, link)
                links += 1
    for variant in VARIANTS:
        assert sha256(REPORT / variant / "result2.xlsx") == sha256(OUT / variant / "result2.xlsx")
    figures = list((REPORT / "figures").glob("*.png"))
    assert len(figures) == 9
    for file in figures:
        assert file.with_suffix(".svg").is_file()
    assert "Ran 41 tests" in (REPORT / "tests.txt").read_text()
    assert re.search(r"(?m)^OK$", (REPORT / "tests.txt").read_text())
    qa = read(REPORT / "visual_qa.json")
    assert qa["status"] == "passed"
    assert qa["snapshot_sha256"] == build["data_sha256"]
    write_json(REPORT / "delivery_verification.json", {"status": "passed", "charts": len(charts),
        "report_sections": 8, "static_figure_pairs": len(figures), "workbooks": 3,
        "resolved_markdown_links": links, "history_experiments": 3, "relative_comparisons": len(comparison),
        "data_sha256": build["data_sha256"], "html_sha256": build["html_sha256"],
        "visual_qa_sha256": sha256(REPORT / "visual_qa.json")})
    print("Delivery verified")


if __name__ == "__main__":
    run()
