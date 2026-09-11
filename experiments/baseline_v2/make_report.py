"""Build a paired offline readout from a suite, retaining all failed cases."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def build(suite):
    lines = ["# v2 配对离线验证", "", "这是自建离线环境的结果，不是官方模拟器演练或正式测试。",
             "每一对使用相同源位置、频道、接收半径、朝向和固定位置误差规则；v1 保持原始默认参数。",
             "两种算法经过同一运行包装器和独立 Benchmark 计时重放。失败案例保留，全清优先。", "",
             f"套件状态：`{suite['status']}`；计划 {suite['planned_pairs']} 对；实际记录 {len(suite['rows'])} 局。",
             f"源码 SHA-256：`{suite['source_sha256']}`。", "",
             "| 题型 | 算法 | 全清且验证通过 / 局数 | 每局平均时间的均值（秒/个） | 清除失败次数均值 | 最后清除后耗时均值（秒） |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    def mean(values):
        return f"{statistics.mean(values):.2f}" if values and all(v is not None for v in values) else "不可用"
    for problem in (3, 4):
        for algorithm in ("v1", "v2"):
            rows = [r for r in suite["rows"] if r["problem"] == problem and r["algorithm"] == algorithm]
            if not rows:
                continue
            avg = mean([r["official_metrics"]["average_localization_clear_time_s"] for r in rows])
            failures = mean([r["diagnostics"]["clear_failure_count"] for r in rows])
            tail = mean([r["v2_diagnostics"]["post_last_clear_time_s"] for r in rows])
            lines.append(f"| Q{problem} | {algorithm} | {sum(r['passed'] for r in rows)}/{len(rows)} | {avg} | {failures} | {tail} |")
    lines += ["", "均值是 mean(Tᵢ/Kᵢ)，含清除后确认成本；非完整日志的均值记为不可用。若有失败，时间不能单独用于宣称改进。",
              "", "| 场景 | v1 全清验证 | v2 全清验证 | v1 秒/个 | v2 秒/个 | 配对时间变化 |", "| --- | --- | --- | ---: | ---: | ---: |"]
    pairs = {}
    for r in suite["rows"]:
        if r.get("source_sha256") != suite["source_sha256"]:
            raise ValueError("Source mismatch inside suite")
        pair = pairs.setdefault(r["scenario"], {})
        if r["algorithm"] in pair:
            raise ValueError("Duplicate algorithm within pair")
        pair[r["algorithm"]] = r
    for name, pair in pairs.items():
        a, b = pair.get("v1"), pair.get("v2")
        if a is None or b is None:
            lines.append(f"| {name} | 未完成配对 | 未完成配对 | — | — | — |")
            continue
        if a["scenario_sha256"] != b["scenario_sha256"]:
            raise ValueError("Refusing to compare different source scenarios")
        x, y = [r["official_metrics"]["average_localization_clear_time_s"] for r in (a, b)]
        delta = f"{(y/x-1)*100:+.1f}%" if a["passed"] and b["passed"] and x and y is not None else "不可比较"
        fmt = lambda v: f"{v:.2f}" if v is not None else "—"
        lines.append(f"| {name} | {a['passed']} | {b['passed']} | {fmt(x)} | {fmt(y)} | {delta} |")
    lines += ["", "负值表示 v2 更快。上述有限场景包含原 v1 种子以及分层和压力案例，不代表官方总体分布，",
              "也没有跨随机案例构造官方配对效果。不同算法程序耗时在本机测得，不预测 Windows HTTP 性能。",
              "", "每局证据位于同目录的场景子文件夹：metadata、events、decisions、offline_truth、metrics。",
              "`verify_artifacts.py` 可独立核对计时、真值包含性和包围圆清除证据。", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    out = args.output or args.suite.with_name("COMPARISON.md")
    out.write_text(build(suite), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
