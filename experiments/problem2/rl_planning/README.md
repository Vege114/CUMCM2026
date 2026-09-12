# exp007：强化学习午夜规划（测速完成，待确认）

独立分支 `codex/q2-rl-planning`，基于 exp006 完成提交 `090bf46`，工作树 `.work/q2-rl-planning`。预测继续读取季节讨论分支归档的 exp004，不重训预测网络。报告模板逐字节同步自 main `7ecd9b9`；正式构建前再次检查更新。

**当前仅完成受限速度试验与单元核验，未正式训练、未全年回放、未生成正式报告或登记成绩。用户确认后才能执行正式矩阵。** 测速模型混用不同历史截止窗口，仅用于速度和物理检查，已丢弃，不得作为年度策略或预训练起点。

## 代码及任务定义

- `environment.py`：因果历史缓存、29 维午夜观测、45 类联合动作、完整日 rollout 及物理执行奖励。9 档有符号电池动作 × 5 档购电裕度；购电以同源历史残差的 0.8 分位为中心。所有 144 槽购电在午夜锁定，规划观测沿计划 SOC 前进，实际 SOC 和实况奖励均不反馈给当天规划。
- `ppo.py`：TensorFlow CPU 单线程 PPO，两层 64 tanh 策略/价值共享网络，clip 0.2，Adam 3e-4，完整日 GAE，检查点包含优化器和 RNG。
- `benchmark.py`：16/32 环境各一次预热与三次短更新，五个日期各三次推理计时；无正式执行参数。源码哈希与原始计时保存到 `data/results/exp007/pilot/benchmark.json`。
- `protocol.proposed.json`：待确认的六组配置，主组三个 RL 种子、纯费用奖励三个种子，全部固定同一 exp004 `no_season/seed42` 预测。
- `tests/test_q2_rl_*.py`：PPO 数值/更新/恢复、真实数据未来篡改、规划信息隔离、极端物理守恒、执行前缀和奖励记账测试。

PPO 原理参考 [Schulman 等原始论文](https://arxiv.org/abs/1707.06347)。电池运行强度以吞吐、EFC 和功率变化记录；真实老化受循环、SOC、温度等因素影响，当前题目缺少电芯老化参数，未推算寿命或真实折旧费，参见 [NREL 电池寿命研究](https://www.nrel.gov/transportation/battery-lifespan.html)。

## 正式预算与因果协议

推荐 32 环境。每组首次 218 次 PPO 更新（1,004,544 步），之后每 14 日只用已完成的最近至多 90 日继续 44 次更新；全年共 23 次继续训练，总计 5,667,840 步。每次更新重新采样 32 个完整日、4 个优化 epoch，batch 512，GAE gamma=1、lambda=0.95。初始只可使用 1 月 9–31 日的 23 条历史周期预测，明确标为冷启动回退；2 月以后使用已发布的冻结预测与已揭晓实况。随机初始 SOC 是训练增强，正式 334 日连续承接真实执行状态，不每日复位。

全年主组预先固定轻惩罚、RL seed42；seed2026/3407 作训练稳定性检查，不能事后选最优种子替换主组。另三个种子去除吞吐、反转和爬坡软罚，保留相同物理约束和 2 kWh 执行死区，以分离额外偏好的费用代价。预算固定，不用 2025 年待评价日的费用选择超参数或停训点。

继承 exp006：SOC 1200–10800 kWh；单向功率≤5000 kW；每槽≤833.333 kWh；单向效率 sqrt(0.9)，往返效率90%。题目的效率措辞可有解释空间，本实验不改变历史计算口径。购电和储能动作有限离散化，PPO 也没有全局最优保证。

奖励为 `-(计划费+5倍紧急费+0.002*吞吐kWh+0.05*非空方向反转+0.0002*|Δ功率kW|)/1000`。前三种电池罚项属于次要偏好；没有额外硬爬坡上限。充放互斥、SOC、功率以及禁止紧急充电由动作结构及执行投影保证。终端训练奖励使用谷价对库存变化估值以缓解日回合边界效应，真实费用不抵扣此值；最后评价日调用方须关闭终端奖励。该日回合近似不等价于严格年度随机最优控制。

正式入口尚未实施。确认后新增可恢复年度 runner、独立核验、工作簿导出与 exp007 报告构建器；不得调用 exp006 runner 改写旧结果。正式入口需特别核验：checkpoint 的观测截止、优化器/RNG 恢复、午夜完整计划哈希、跨日 SOC/方向/上一槽功率、最后日终端估值、任何非有限训练/验证错误立即留痕停止。

## 已执行测速与检查

从本工作树运行（复用根项目只读环境）：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/exp007-matplotlib XDG_CACHE_HOME=/private/tmp/exp007-cache \
/Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python -u \
  -m experiments.problem2.rl_planning.benchmark --repeats 3

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/exp007-matplotlib XDG_CACHE_HOME=/private/tmp/exp007-cache \
/Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python \
  -m unittest discover -s tests -p 'test_q2_rl_*.py' -v
```

测速入口拒绝覆盖已归档 benchmark.json。复测使用独立副本和空 pilot 路径，先保留本次证据。不能为了查看结果重新运行测速，更不能把 `speed_only_day*.npz` 当作正式 result2.xlsx 数据。

[测速及待确认预算](../../../reports/experiments/exp007/preflight.md) · [main 模板与历史比较契约](../../../reports/experiments/exp007/report_contract.md)
