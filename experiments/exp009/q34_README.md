# Q3 / Q4-3：退款结算与正式策略冷启动

依据 `Xelatex/第四问冲突清单.md` 的 F10、F12 人工决定，本次在提交 `459cdb2` 选定的 Q3/Q4-3 方法上修改下调结算，并从 2025 年 1 月 1 日的 6000 kWh 接入自身调度；未读取 exp002 预热状态。冻结的 exp008 源码和历史轨迹保留供来源核对。

## 计算口径

- 原午夜计划为 `original`，每个十分钟槽最后一个有效版本为 `final`。后续发布不覆盖已经执行的槽。
- 最终账单为 `p * (original + 1.5 * max(final-original,0) - 0.5 * max(original-final,0) + 5 * emergency)`。`fees[...,2]` 为负数的下调退款。
- LP 中购电 `q >= 0`，与非负变量 `up, down` 满足 `q - up + down = original`；增量目标为 `1.5*p*up - 0.5*p*down`。同时增加上调和下调会增加成本，故最优解自动采用正部。
- 每次预测只规划至当天 24:00；代表路径数 7，发布时刻为 0、6、12、18 时。下一发布以前的短缺权重为 5，以后为 1.5；真实紧急账单始终按 5 倍结算。原 LP 的其他目标、库存价值和实际执行规则保持不变。
- 一月使用正式预测适配器已有的因果周期冷启动：首日附件 1，随后负荷优先上周同槽、不足一周用昨日，光伏用昨日。Q3/Q4-3 从首日 0 时即接入合法发布的官方光伏；2 月起使用已验收的单 HGB、Ridge28、非递归半记忆缓存。无完整历史日时采用零误差单路径。1 月 2 日 0 时尚无满足滞后设计的价格回归训练行，按原价格先验采用已观测的昨日价格。
- 电池真实状态跨日连续，不对每日起点重置；末日取消库存估值，不额外施加末端等式。

## 输出

`data/results/exp009/q3/` 和 `q4_3/` 中，`dispatch_365.npz` 保存 365 日完整运行；`warmup.npz` 保存 1 月；`dispatch_3.npz` / `dispatch_4-3.npz` 保存附件要求的 2 月 1 日至 12 月 31 日 334 日。`summary.json` 为正式评价汇总，`verification_365.json` 同时保存 365 日账单和物理核验。

`versions` 的形状为 `(day,4,144)`，记录每次发布后的购电版本；其已执行部分保持不变。`states` 保存每个自然日的 145 个边界状态；`charge`、`discharge` 是交流侧 kWh，`states` 是电芯内部 kWh，功率为电量乘 6。

`result3.xlsx` 与 `result4-3.xlsx` 使用题目模板，保留四张工作表；原计划工作表的费用列只填原计划费，调整工作表的费用列填包含退款与紧急购电的最终总费。调整购电量填写调整后总常规电量。模板十分钟列名修正为与右端点原始数据对应的 `00:00-00:10` 至 `23:50-24:00`。模板及结果均无公式，所有合计由原精度数组计算后写入；导出后重新读取逐单元格核验。

## 命令

首次运行（已有完整结果时拒绝覆盖，可用 `--label replay` 写到独立复验目录）：

```bash
.venv/bin/python -m experiments.exp009.q34_run --scenario 3
.venv/bin/python -m experiments.exp009.q34_run --scenario 4-3
.venv/bin/python -m experiments.exp009.q34_run --causality
.venv/bin/python -m experiments.exp009.q34_export --forecast-metrics
.venv/bin/python -m experiments.exp009.q34_assets --question all
```

`protocol.json` 固定输入 SHA-256、模型缓存哈希、源文件哈希、参数、种子和命令；`completion.json` 为实测墙钟时间及求解累计时间。独立 P1/P2 验收回执由未参与编写的审计 Agent 生成，不能以作者自检替代。

`q34_assets` 仅由已完成、已核验的轨迹重建正文图表，不执行策略求解；问题四需等待 `q4_2/summary.json` 标记 365 日完成。图表输出到 `Xelatex/final_results/`，图像为相同尺寸的 PDF、320 dpi PNG、SVG 及灰度预览。费用使用原精度合计后显示两位小数，电量显示四位小数。价格图仅比较同一 48096 槽的午夜预测，问题三预测误差表仅比较不重叠的执行窗口。
