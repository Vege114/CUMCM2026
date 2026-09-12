"""Build the exp005 D4-A evidence report; never changes the approved model."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.problem2.exp003.data import Data  # noqa: E402
from experiments.problem2.stochastic_lp.model import Config, State, Tree, solve_tree  # noqa: E402

OUT = ROOT / "reports/experiments/exp005"
RUN = ROOT / "data/results/exp005/beta0-boundary"
EVIDENCE = OUT / "evidence"
DAY = RUN / "2025-02-01"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def read(path):
    return json.loads(path.read_text())


def rows(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(t):
    return f"{t // 6:02}:{t % 6 * 10:02}—{(t + 1) // 6:02}:{(t + 1) % 6 * 10:02}"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--complete", action="store_true", help="failure report authoring complete, not policy certification")
    parser.add_argument("--build", action="store_true", help="build and export the existing report app")
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    failure, protocol = read(RUN / "certification_failure.json"), read(RUN / "protocol.json")
    z = dict(np.load(DAY / "execution-081.npz"))
    prefix = dict(np.load(DAY / "executed_prefix.npz"))
    midnight = dict(np.load(DAY / "midnight.npz"))
    cfg = Config(**protocol["config"])
    state = State(**failure["initial_state"])
    data = Data()
    tree = Tree(z["time"], z["parent"], z["probability"], z["supply_kw"], 63)
    # Same input, same two objectives. Preserve first-layer arrays for inspection.
    replayed = solve_tree(tree, data.fixed_price[81:], state, cfg, z["grid"])
    max_difference = max(float(np.max(np.abs(replayed[key] - z[key]))) for key in
                         ["grid", "charge", "discharge", "emergency", "end_soc", "net_power_kw"])
    if max_difference > cfg.tolerance:
        raise RuntimeError("exact failed-window reproduction differs; investigate before reporting")
    np.savez_compressed(EVIDENCE / "failed_window_both_layers.npz",
                        **{k: v for k, v in replayed.items() if k != "metadata"},
                        time=tree.time, parent=tree.parent, probability=tree.probability,
                        supply_kw=tree.supply_kw)
    dump(EVIDENCE / "failed_window_both_layers.json", replayed["metadata"])
    for file in [RUN / "certification_failure.json", RUN / "protocol.json", RUN / "status.json",
                 DAY / "executed_prefix.npz", DAY / "executed_prefix_logs.json", DAY / "scenarios.npz",
                 DAY / "midnight.npz", DAY / "midnight.json"]:
        shutil.copy2(file, EVIDENCE / file.name)
    eta = np.sqrt(.9)
    initial = protocol["initial_state"]
    power = prefix["net_power_kw"]
    max_prefix_balance = float(np.abs(prefix["grid"] + prefix["actual"][:, 1] / 6
                                    + prefix["discharge"] + prefix["emergency"]
                                    - prefix["actual"][:, 0] / 6 - prefix["charge"] - prefix["surplus"]).max())
    soc_error = float(np.abs(np.diff(prefix["states"]) - eta * prefix["charge"] + prefix["discharge"] / eta).max())
    previous = np.r_[initial["previous_power_kw"], power[:-1]]
    actual_visible = data.actual[31 * 144:31 * 144 + 81]
    checks = [
        {"check": "失败窗口原样复算最大差", "value": max_difference, "limit": 1e-6, "unit": "数值原单位", "passed": max_difference <= 1e-6},
        {"check": "81段供需平衡最大残差", "value": max_prefix_balance, "limit": 1e-6, "unit": "kWh", "passed": max_prefix_balance <= 1e-6},
        {"check": "81段SOC递推最大残差", "value": soc_error, "limit": 1e-6, "unit": "kWh", "passed": soc_error <= 1e-6},
        {"check": "81段实际供需与附件2最大差", "value": float(np.abs(actual_visible - prefix["actual"]).max()), "limit": 0., "unit": "kW", "passed": bool(np.array_equal(actual_visible, prefix["actual"]))},
        {"check": "81段普通计划变更最大量", "value": float(np.abs(prefix["grid"] - midnight["grid"][:81]).max()), "limit": 0., "unit": "kWh", "passed": bool(np.array_equal(prefix["grid"], midnight["grid"][:81]))},
        {"check": "81段最大充放电重叠", "value": float(np.minimum(prefix["charge"], prefix["discharge"]).max()), "limit": 1e-6, "unit": "kWh", "passed": bool((np.minimum(prefix["charge"], prefix["discharge"]) <= 1e-6).all())},
        {"check": "81段最大相邻功率变化", "value": float(np.abs(power - previous).max()), "limit": 1000., "unit": "kW", "passed": bool(np.abs(power - previous).max() <= 1000 + 1e-6)},
    ]
    if not all(r["passed"] for r in checks):
        raise RuntimeError("executed-prefix independent audit failed")
    for stage, name in [("", "第二层"), ("first_", "第一层")]:
        c, d, e, w, soc, p = [replayed[stage + k] for k in
                              ["charge", "discharge", "emergency", "surplus", "end_soc", "net_power_kw"]]
        parent_soc = np.array([state.soc if a < 0 else soc[a] for a in tree.parent])
        parent_p = np.array([state.previous_power_kw if a < 0 else p[a] for a in tree.parent])
        residual = max(float(np.abs(z["grid"][tree.time] + tree.supply_kw[:, 1] / 6 + d + e
                                    - tree.supply_kw[:, 0] / 6 - c - w).max()),
                       float(np.abs(soc - parent_soc - eta * c + d / eta).max()))
        checks += [{"check": name + "失败窗口能量/SOC最大残差", "value": residual, "limit": 1e-6, "unit": "kWh", "passed": residual <= 1e-6},
                   {"check": name + "最大相邻功率变化", "value": float(np.abs(p - parent_p).max()), "limit": 1000., "unit": "kW", "passed": bool(np.abs(p - parent_p).max() <= 1000 + 1e-6)}]
    checks.append({"check": "第二层未来节点最大充放电重叠", "value": failure["overlap_kwh"],
                   "limit": 1e-6, "unit": "kWh", "passed": False})
    tests = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "unittest", "discover", "-s", "tests",
                            "-p", "test_q2_stochastic_lp.py", "-v"], cwd=ROOT, text=True, capture_output=True, check=False)
    (EVIDENCE / "tests.txt").write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise RuntimeError("model tests failed")
    dump(EVIDENCE / "independent_audit.json", {"checks": checks, "test_suite_passed": True,
                                               "physical_certification": False, "completed_days": 0,
                                               "executed_intervals": 81, "stopped_before_slot": 81})

    prefix_frame = pd.DataFrame({"slot": np.arange(81), "hour": np.arange(81) / 6,
                                "interval": [interval(t) for t in range(81)],
                                "grid_kwh": prefix["grid"], "charge_kwh": prefix["charge"],
                                "discharge_kwh": prefix["discharge"], "emergency_kwh": prefix["emergency"],
                                "start_soc_kwh": prefix["states"][:-1], "end_soc_kwh": prefix["states"][1:],
                                "net_power_kw": prefix["net_power_kw"],
                                "planned_cost": prefix["fees"][:, 0], "emergency_cost": prefix["fees"][:, 1]})
    window = []
    for stage, label in [("first_", "第一层·费用"), ("", "第二层·平稳性")]:
        for t in range(63):
            window.append({"stage": label, "slot": t + 81, "hour": (t + 81) / 6,
                           "interval": interval(t + 81), "grid_kwh": replayed["grid"][t],
                           **{key + "_kwh": float(replayed[stage + key][t]) for key in
                              ["charge", "discharge", "emergency", "surplus", "end_soc"]},
                           "power_kw": float(replayed[stage + "net_power_kw"][t]),
                           "overlap_kwh": float(min(replayed[stage + "charge"][t], replayed[stage + "discharge"][t]))})
    window_frame = pd.DataFrame(window)
    stages = [
        {"stage": "第一层·费用", "emergency_cost": replayed["metadata"]["first_cost"],
         "tv_kw": replayed["metadata"]["first_expected_tv_kw"],
         "max_overlap_kwh": replayed["metadata"]["stages"][0]["max_overlap_kwh"]},
        {"stage": "第二层·平稳性", "emergency_cost": replayed["metadata"]["second_cost"],
         "tv_kw": replayed["metadata"]["expected_tv_kw"],
         "max_overlap_kwh": failure["overlap_kwh"]}]
    node = window_frame[(window_frame.stage == "第二层·平稳性") & (window_frame.slot == 85)].iloc[0]
    minmax = [{"metric": "开始储电量", "value": float(replayed["end_soc"][3]), "unit": "kWh"},
              {"metric": "充电量", "value": float(node.charge_kwh), "unit": "kWh"},
              {"metric": "放电量", "value": float(node.discharge_kwh), "unit": "kWh"},
              {"metric": "重叠量", "value": float(node.overlap_kwh), "unit": "kWh"},
              {"metric": "结束储电量", "value": float(node.end_soc_kwh), "unit": "kWh"},
              {"metric": "净充电功率", "value": float(node.power_kw), "unit": "kW"}]
    for filename, frame in [("executed_prefix.csv", prefix_frame), ("failed_window.csv", window_frame),
                            ("solver_stages.csv", pd.DataFrame(stages)), ("audit.csv", pd.DataFrame(checks))]:
        frame.to_csv(EVIDENCE / filename, index=False, encoding="utf-8-sig")
    source_paths = [ROOT / "问题二_场景随机线性规划与结算_数学推导讨论稿.md",
                    ROOT / "data/results/exp004/predictions.npz", ROOT / "data/results/exp004/prediction_manifest.json",
                    ROOT / "data/results/exp004/no_season/warmup_2.npz",
                    *[(ROOT / "data/raw" / name) for name in data.hashes],
                    ROOT / "experiments/problem2/linear_planning/model.py",
                    ROOT / "tests/test_q2_stochastic_lp.py",
                    *sorted((ROOT / "experiments/problem2/stochastic_lp").glob("*.py"))]
    dump(EVIDENCE / "source_hashes.json", {str(p.relative_to(ROOT)): digest(p) for p in source_paths})

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
                         "font.size": 11, "svg.fonttype": "none"})
    def savefig(name):
        plt.savefig(figures / (name + ".png"), dpi=180, bbox_inches="tight")
        plt.savefig(figures / (name + ".svg"), bbox_inches="tight")
        plt.close()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, sharey=True)
    for ax, label in zip(axes, ["第一层·费用", "第二层·平稳性"]):
        subset = window_frame[(window_frame.stage == label) & (window_frame.slot <= 99)]
        ax.plot(subset.hour, subset.charge_kwh, "o-", markersize=3, color="#258a79", label="充电量")
        ax.plot(subset.hour, subset.discharge_kwh, "o-", markersize=3, color="#bf6548", label="放电量")
        ax.axvspan(85 / 6, 86 / 6, color="#e8bd90", alpha=.28)
        ax.set_title(label, loc="left", fontsize=12)
        ax.set_ylabel("kWh / 十分钟")
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="upper right", frameon=False)
    axes[1].annotate("14:10—14:20\n同时充电740.6564、放电566.5938 kWh",
                     xy=(85 / 6, node.charge_kwh), xytext=(14.9, 675), fontsize=10,
                     arrowprops={"arrowstyle": "->", "color": "#715139"})
    axes[1].set_xlabel("2025-02-01时刻 / 小时；均为13:30求解的未来规划")
    fig.suptitle("费用层无重叠，平稳性层返回一处重叠", fontsize=15, x=.1, ha="left")
    fig.tight_layout()
    savefig("failed-window")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(np.arange(82) / 6, prefix["states"], color="#258a79", lw=2)
    axes[0].axhline(10800, ls="--", color="#888", lw=1)
    axes[0].axhline(1200, ls="--", color="#888", lw=1)
    axes[0].set_ylabel("内部储电量 / kWh")
    axes[1].step(np.arange(81) / 6, power, where="post", color="#416c9b")
    axes[1].set_ylabel("净充电功率 / kW")
    axes[1].set_xlabel("2025-02-01时刻 / 小时；仅含已执行的00:00—13:30")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("停止前81段已执行轨迹：SOC与功率保持连续", fontsize=15, x=.1, ha="left")
    fig.tight_layout()
    savefig("executed-prefix")
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.barh([s["stage"] for s in stages], [s["max_overlap_kwh"] for s in stages], color=["#258a79", "#bf6548"])
    ax.set_xlim(0, 700)
    for i, s in enumerate(stages):
        ax.text(s["max_overlap_kwh"] + 10, i, f'{s["max_overlap_kwh"]:.4f}', va="center")
    ax.set_xlabel("13:30失败窗口中的最大重叠量 / kWh")
    ax.set_title("重叠诊断与LP可行性是两项不同的检验", loc="left")
    savefig("overlap-by-layer")

    total_plan_cost = float(np.sum(midnight["grid"] * data.fixed_price))
    prefix_emergency_cost = float(prefix["fees"][:, 1].sum())
    sections = [
        ("结论与成绩", f"## 1. 结论与成绩\n\n本轮交付为D4-A失败证据报告。原模型保持不变，β=0；用户在看到失败后明确要求仅整理证据。正式334日评价没有完成，不生成result2.xlsx，不报告全年费用或节约率。\n\n2025年2月1日13:30的滚动求解，在未来14:10—14:20节点返回同时充电 **{node.charge_kwh:.4f} kWh**、放电 **{node.discharge_kwh:.4f} kWh**。该窗口第一层无重叠，第二层有一处重叠。当次当前节点没有重叠，停止前81段实际执行也没有重叠。发现后在执行13:30—13:40动作之前停止认证。", "stages"),
        ("题目指标及信息边界", "## 2. 题目指标及信息边界\n\n购电、充电、放电、未利用量的单位均为kWh，功率输入乘1/6小时转换。单向效率均为√0.9，SOC范围1200—10800 kWh，单向充放电上限5000 kW，净功率相邻变化上限1000 kW。\n\n午夜第一层最小化计划费加信息树预期紧急费，第二层在第一层最优费用的1.001倍加0.0001元以内最小化预期净功率总变差。日内固定普通计划，仅最小化剩余预期紧急费，再优化平稳性；只执行当前动作。D5-A允许紧急电服务充电。\n\n重叠量定义为min(充电量,放电量)，超过0.000001 kWh触发D4-A。重叠不是本连续LP的一条约束，所以“LP最优且残差很小”和“轨迹不能作为单向电池动作”可以同时成立。当前段供需采用区间内计量反馈的十分钟平均近似，不能解读为区间开始前已知未来平均值。", "audit"),
        ("数据与时间验证", "## 3. 数据与时间验证\n\n使用附件1固定电价、附件2负载和光伏实际值、exp004/no_season_seed_42冻结预测。没有重训或读取附件3、4。预测分支提交为b42168d5271097762953f38d472d0ef5fe1a908d；预测档案SHA-256与其清单核对一致。\n\n预测档案从2月1日开始，因此首日没有已完成的历史预测误差供体，信息树明确退化为单条点预测路径。本次失败发生在引入多个历史场景之前，不能归因于16条路径抽样或树分辨率。\n\n共同一月预热提供2月1日初始SOC1421.7991105135516 kWh、前一段净充电功率172.76 kW。真实回放中的当前供需与附件2逐项一致；未来实际值不进入午夜函数。每个原始输入和实现源文件的哈希记录在证据目录。", "audit"),
        ("逐步技术讲解", "## 4. 逐步技术讲解\n\n数据流：冻结午夜预测 → 历史实际减当时最终预测 → 整日联合误差路径 → 非负裁剪与历史日照掩码 → 前缀信息树 → 午夜两层LP → 固定购电计划的逐段两层滚动LP → 当前动作与实际账单。\n\n树中每个节点有唯一父节点、一组代表负载/PV和一组共享动作。平衡式为g+v+d+e=ℓ+c+w；SOC递推为E后=E前+√0.9c−d/√0.9；净功率为P=6(c−d)。节点概率在同一时段加总为1，计划费只计算一次，紧急费和总变差乘节点概率。\n\n每个父节点只允许按当前可见的标准化前缀继续二分；当前代表供需取组内均值，不把不同当前供需强塞进同一补救变量。日内当前根节点采用实际计量，未来仍用既有场景，权重不更新。费用和平稳性顺序求解，始终没有第三层、整数互斥或终端储能奖励。", "settings"),
        ("实验设置", "## 5. 实验设置\n\n用户确认的实现参数：最近最多56个完整历史日，最多16条路径，等权无放回抽样，随机种子42；06:00、12:00、18:00的当前反馈段按前缀二分，每个子组至少2条路径；日内不更新权重。午夜δ与执行δ均0.001，费用容差0.0001元，β=0。\n\n求解器为SciPy/HiGHS连续LP，每层必须取得最优状态并检查残差。首日午夜求解一次，两层；随后执行了81个区间，并在第82次滚动求解时发现未来节点重叠。D4检查覆盖第二层返回的全部节点，不能只看当前动作。\n\n没有实施曾询问的β=0.001检验，没有放松硬爬坡，没有净化充放电量。小树分支和未来扰动属于实现测试，不代表多日真实场景效果已被验证。", "settings"),
        ("结果及失败案例", f"## 6. 结果及失败案例\n\n13:30窗口的第一层剩余预期紧急费为{stages[0]['emergency_cost']:.4f}元，总变差{stages[0]['tv_kw']:.4f} kW；第二层剩余预期紧急费为{stages[1]['emergency_cost']:.4f}元，总变差{stages[1]['tv_kw']:.4f} kW。第二层费用预算为0.0001元。这里的费用和总变差均属于63段规划窗口，不是实际全天结算。\n\n重叠节点开始SOC为{minmax[0]['value']:.4f} kWh，结束SOC为{node.end_soc_kwh:.4f} kWh，净充电功率为{node.power_kw:.4f} kW。它满足供需平衡、SOC和硬爬坡，仍返回充放电重叠。该证据证明求解器返回的这条第二层松弛轨迹存在物理偏差；尚未证明重叠是所有第二层最优解必需的。\n\n停止前实际执行范围为00:00—13:30，共81段。午夜已锁定的全天计划费为{total_plan_cost:.4f}元；已执行81段的实际紧急费为{prefix_emergency_cost:.4f}元。两者不可称作完整日费，因为余下63段实际紧急量尚未回放。3月20日、6月21日、9月23日、12月21日结果均未生成。", "stages"),
        ("历次指标和技术路线对比", "## 7. 历次指标和技术路线对比\n\nexp004提供冻结预测并沿用旧执行器作预测实验；本轮接入同一预测，目标是检验推导稿的两层连续规划与滚动执行。首日信息树退化成点路径，便已触发D4-A。\n\n本轮没有完整日费或正式评价期成绩，因此不与exp001—exp004的全年成绩计算改善率、不作排名。该停止也不能证明exp004预测变差，更不能据此判断场景模型的全年费用。原有实验记录和报告没有修改，本轮不进入完整实验成绩登记。", None),
        ("复现说明", "## 8. 复现说明\n\n运行 `.venv/bin/python -m unittest discover -s tests -p test_q2_stochastic_lp.py -v` 检查11项模型测试。运行 `.venv/bin/python -m experiments.problem2.stochastic_lp.run --days 2 --out .work/exp005-reproduce` 可复现首日停止，输出目录必须不存在。\n\n运行 `.venv/bin/python reports/build_report_exp005.py` 重建证据、静态图和正文；构建器在完全相同输入和原模型下复算失败窗口，保存两层数组并核对原结果，没有改变目标或约束。网页使用与既有报告相同的Data共享运行时，数据完整内嵌到离线HTML。\n\n证据包包括失败记录、固定协议、午夜144段计划、停止前81段实际轨迹、13:30失败窗口的两层完整数组、CSV明细、独立审计、测试日志和源文件哈希。失败报告整理完成不等于购电策略通过认证。", "audit"),
    ]
    queries = {}
    labels = {"dt": "区间时长 Δt / 小时", "eta_rt": "往返效率", "soc_min": "SOC下限 / kWh", "soc_max": "SOC上限 / kWh",
              "power_max": "单向功率上限 / kW", "ramp_kw": "相邻段功率变化上限 / kW", "delta": "午夜费用让步比例 δ",
              "delta_exec": "执行费用让步比例 δ_exec", "epsilon_cost": "费用容差 / 元", "beta": "周转权重 β",
              "time_limit": "单层求解时限 / 秒", "tolerance": "数值与重叠判断容差"}
    query_data = {"prefix": (rows(prefix_frame), "已执行81段，00:00—13:30；SOC为各段末值，费用仅为该前缀的区间费用。", "executed_prefix.csv"),
                  "window": (rows(window_frame), "2025-02-01 13:30求解的63段未来规划；第一层与第二层分开保存，均不是实际执行。", "failed_window.csv"),
                  "stages": (stages, "同一次13:30窗口两层最优解，费用为剩余预期紧急费，总变差含窗口开始的真实上一段功率。", "solver_stages.csv"),
                  "node": (minmax, "第二层14:10—14:20节点；重叠为min(c,d)，单位按行给出。", "failed_window.csv"),
                  "audit": (checks, "从保存数组独立核对，重叠诊断与线性约束残差分别报告。", "audit.csv"),
                  "settings": ([{"parameter": labels[k], "value": v} for k, v in protocol["config"].items()], "本轮原始固定参数；β=0。", "protocol.json")}
    for identity, (values, definition, filename) in query_data.items():
        queries[identity] = {"rows": values, "source": {"label": "exp005 实测边界证据",
                                "files": ["evidence/" + filename], "filters": ["只含已执行前缀或明确标记的规划窗口"],
                                "metricDefinitions": [{"label": identity, "definition": definition}],
                                "evidenceFlow": [{"title": "读取冻结证据", "detail": "读取同目录证据文件；原模型复算与独立核对见build_report_exp005.py。"}]}}
    snapshot = {"title": "第二问：平稳性层出现充放电重叠", "surface": "report", "status": "reviewed",
                "buildStatus": "complete" if args.complete else "creating", "generatedAt": datetime.now(UTC).isoformat(),
                "report": {"asOf": "2025-02-01"}, "filters": [], "queries": queries,
                "sections": [{"title": title, "markdown": markdown, "queryId": q} for title, markdown, q in sections]}
    existing = OUT / "app/src/data.json"
    if existing.exists():
        snapshot["id"] = read(existing)["id"]
        dump(existing, snapshot)
    dump(OUT / "reviewed.json", snapshot)
    figure_sections = {1: "overlap-by-layer", 6: "failed-window"}
    body = ["# exp005：问题二连续LP的D4-A失败证据\n\n沿用八节实验报告模板；原模型β=0，正式评价停止。"]
    for i, (_, markdown, _) in enumerate(sections, 1):
        if i in figure_sections:
            markdown += f"\n\n![{figure_sections[i]}](figures/{figure_sections[i]}.png)"
        if i == 6:
            markdown += "\n\n![停止前实际轨迹](figures/executed-prefix.png)"
        (OUT / f"section-{i}.md").write_text(markdown + "\n")
        body.append(markdown)
    (OUT / "report.md").write_text("\n\n".join(body) + "\n")
    if args.build:
        node = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
        plugin = Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2/scripts/data-app.mjs"
        app = OUT / "app"
        subprocess.run([str(node), str(plugin), "build", "--project-dir", str(app), "--separate-data"], check=True)
        offline = app / ".data-app-offline/exports/report.html"
        subprocess.run([str(node), str(plugin), "export-offline", "--project-dir", str(app), "--output", str(offline)], check=True)
        shutil.copy2(offline, OUT / "report.html")
        shutil.copytree(EVIDENCE, app / "dist/evidence", dirs_exist_ok=True)
    print(json.dumps({"report": str(OUT), "audits": len(checks), "tests_passed": True,
                      "reproduction_max_difference": max_difference,
                      "stage_metrics": stages, "prefix_intervals": len(prefix_frame)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
