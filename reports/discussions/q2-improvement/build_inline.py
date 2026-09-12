"""Render the compact Visualize overview from the same discussion evidence."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def build(output=None):
    evidence = json.loads((HERE / "evidence.json").read_text())
    labels = {"v2_primary": "v2正式", "exp003_uncalibrated": "未校准网络",
              "exp003_primary": "当前方案", "periodic": "周期基线"}
    payload = {
        "scope": "附件2：2025年1—12月实测数据；费用：2—12月334天、同物理与计费口径。原始曲线为事后观测；当前方案采用周期预测，未优于周期基线。",
        "monthly": evidence["raw"]["monthly"],
        "price": evidence["tariff"]["price_yuan_per_kwh"],
        "costs": [{key: case[key] for key in ("total_cost", "planned_cost", "emergency_cost", "note")}
                  | {"label": labels[case["id"]]} for case in evidence["comparisons"]],
    }
    template = (ROOT / "reports/templates/q2-discussion/inline-overview.html").read_text()
    assert template.count("@@DATA@@") == 1
    fragment = template.replace("@@DATA@@", json.dumps(payload, ensure_ascii=False, allow_nan=False))
    destination = Path(output) if output else HERE / "inline-overview.html"
    destination.write_text(fragment)
    assert destination.stat().st_size < 1_000_000
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    build(parser.parse_args().output)
