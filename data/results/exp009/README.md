# 正式结果：退款与全年冷启动修订

本轮依据全文冲突清单的人工决定，以 exp008 的已验收算法及预测发布值为计算基线（459cdb2），执行两项主结果变更：

1. Q3、Q4-3开放下调，按取消部分退回原款一半求解及结算，保持情景LP、预测校准、执行规则和其余参数。
2. Q2、Q3、Q4-2、Q4-3各自从2025年1月1日6000 kWh开始，以本策略的周期冷启动处理历史不足，再接入已验收树模型；不再继承exp002调度预热状态。

全年连续运行365日，官方结果模板只保存2025年2月1日至12月31日334日。`dispatch_365.npz`（或相应365日归档）供完整轨迹核对；每问目录的`summary.json`标明评价区间，不能把334日费用写成365日账单。Q1采用正式两层MIP第二层典型日轨迹。

## 结果与复现

正式工作簿为当前目录下`result1.xlsx`、`result2.xlsx`、`result3.xlsx`、`result4-2.xlsx`、`result4-3.xlsx`。正式表图分别由Q1第二层JSON、新年度NPZ生成。

在仓库根目录运行，Python 3.12及依赖版本见`requirements.txt`：

```bash
python scripts/prepare_exp009_support.py
python scripts/verify_exp009_delivery.py
```

ZIP为满足20 MB大小要求，仅省略可由完整365日数组精确切出的四份一月/评价期副本；准备命令恢复它们并核对原SHA-256。Git保留全部原切片，已有且哈希正确的文件不会重写。全部工作簿及完整年度数组始终保留。

该命令只复核现有结果与来源，不重新训练或求解。重新调度使用独立输出位置；若已存在同名完整复算结果，更换输出目录或label，保留原结果：

Q2/Q4-2的原始`verification.json`沿用了历史校验模块，其中`goal`是旧实验相对exp006节约8%的开发目标，与本轮人工决定和物理可行性验收不同。保留该历史字段的真实值；本轮统一验收结论以`delivery_verification.json`为准。

```bash
python Xelatex/code/q1_reproduce.py --quick --out q1_output
python -m experiments.exp009.q12_run --scenario 2 --days 365 --out data/results/exp009/reproduced/q2
python -m experiments.exp009.q12_run --scenario 4-2 --days 365 --out data/results/exp009/reproduced/q4_2
python -m experiments.exp009.q34_run --scenario 3 --days 365 --label reproduction
python -m experiments.exp009.q34_run --scenario 4-3 --days 365 --label reproduction
```

Q1最优轨迹存在多解时，以归档的第二层完整轨迹为正式输出；不能重求后混用其他层或不同轨迹的分时表。年度MIP有5秒时间预算，不同硬件可能得到不同可行轨迹，重求结果需另存并复核，不能以目标gap代替实际gap。

预测缓存由exp008的HGB/ExtraTrees训练与校准代码产生，本轮读取它们并重新运行调度，没有重新训练所有月模型。历史训练及预测来源见`data/results/exp008/forecast_absolute_hgb/`和`forecast_hgb_extra_trees_half/`中的provenance和审计文件；本轮文件、原始输入及实际导入闭包SHA-256由`复现清单.json`列出。

## 字段与单位

| 字段 | 含义 |
|---|---|
| original | 午夜原计划，交流侧kWh |
| final | 对应槽最后有效版本的常规购电总量，交流侧kWh；不与original重复相加 |
| charge / discharge | 实际交流侧充/放电量，kWh |
| states | 电池内部储电量，每日145个槽边界，kWh |
| emergency | 实际紧急购电量，kWh |
| surplus | 总未利用电量，kWh，未按光伏/已付购电拆分 |
| price | 用于结算的基础实际电价，元/kWh |
| versions | Q3/Q4-3的0/6/12/18时计划版本，已执行槽保持锁定 |
| fees | 槽费用各分量；Q3/Q4-3第三分量为负退款，其他组无调整费 |

槽数组下标0对应00:00—00:10；指定10:00—10:10取下标60，对应附件右端点10:10。四小时块取半开区间`[24*b:24*(b+1)]`。Q1原始JSON的`c,d`是kW，导出时除以6；年度`charge,discharge`已经是kWh。工作簿显示金额2位、电量4位，但保存未舍入数值。汇总比例从未舍入总额计算。

既有Q1十五组敏感性与Q2固定执行压力资料使用原记录注明的模型和轨迹，不冒充本轮整管线重训练实验。旧Q3严格MIP和旧Q4冻结工作簿保留为历史资料，当前主稿不会读取它们。

## 论文及构建

唯一入口为`Xelatex/数模通用模板.tex`。完整源程序在`Xelatex/code/final/`按原目录结构复制并校验哈希后列入附录。AI工具使用详情源文件为`Xelatex/AI工具使用详情.tex`，PDF在本地编译后收入支撑包。所有TeX编译输出和中间文件均被Git忽略，原始题目PDF及绘图PDF不受影响。

[全文人工决定执行记录](../../../Xelatex/全文决定执行记录.md)列出本轮核验与尚按用户安排暂缓的篇幅优化。`支撑材料.zip`由`python scripts/finalize_exp009.py`整理，仅为本地生成包；模型与可复核数据分别版本管理。
