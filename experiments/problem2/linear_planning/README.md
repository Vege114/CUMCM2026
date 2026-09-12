# 问题二：独立预测接口下的两层连续线性规划

分支：`codex/q2-linear-planning`。基于 `origin/codex/q2-optimization` 的
`dd8a6fd`，不改 exp003、neural_v2、预测模型或问题一论文，不在本次合并。

## 本次决定与实现边界

用户在本分支明确决定：第二问取消第三层购电量优化，不加入充放电互斥约束，
尽量保留第二层。由此采用 **两次连续 LP**，`scipy.optimize.linprog(method="highs")`。
没有整数变量、充放电模式变量或第三次求解。

问题一第二层中，净功率总变差及1000 kW/10分钟变化限制保留；
启动次数、最短运行/关闭和短段缺额不能原样转成连续LP，当前不纳入。
这是一项明确的模型范围调整，不能宣称完整继承问题一的全部运行条件。

## 变量、物理条件与单位

队友提供 `forecast[t,0:2] = [负载功率, 光伏功率]`，单位kW。
全天144段；内部乘 `dt=1/6 h` 转成kWh。电价是附件一已知分时价格，元/kWh。

- `g[t]`：午夜锁定的计划购电量；
- `c[t], d[t]`：交流侧充、放电量；
- `e[t]`：紧急购电量（仅执行模型开放）；
- `w[t]`：未利用电量；
- `E[t]`：区间开始时的电池内部储电量；
- `P[t]=(c[t]-d[t])/dt`：交流侧净充电功率，正为充电。

能量平衡：

`g + PV*dt + d + e = Load*dt + c + w`。

储电递推：

`E[t+1] = E[t] + sqrt(0.9)*c[t] - d[t]/sqrt(0.9)`。

储电范围1200—10800 kWh；`0 <= c,d <= 5000*dt`；`g,e,w>=0`。
相邻功率满足 `abs(P[t]-P[t-1])<=1000 kW`，通过两条线性不等式表达。
日初传入真实SOC与上一段功率；没有日末强制6000、日首尾相等或循环索引。

## 凌晨两层优化

第一层：`C* = min sum(p*g)`。点预测基线令 `e=0`，满足预测需求。
该模型不声称优化了真实全年总费，也没有直接预估预测误差的期望紧急费用；
只有输入点预测时，这是可核验的基础模型。以后可接入误差路径或历史校准备用，
不能把后来观察到的真实供需作为午夜输入。

第二层：在 `sum(p*g) <= (1+0.001)*C* + 1e-4` 下最小化

`F2 = sum(abs(P[t]-P[t-1]))/(2*Pmax*T)`

`     + beta*sum(c[t]+d[t])/(2*Pmax*dt*T)`。

绝对值通过非负辅助变量和正负两条下界线性化。
`beta=0.001` 是显式可配置的第二层小权重，用于抑制无意义能量周转；
不是电池寿命费用、不是已校准参数、不是第三层最小购电量，也不是互斥证明。
如需只优化总变差，可将其设为0，但仍须检查重叠。
费用与平稳性是顺序求解，不是混合成一个加权第一目标。

两个阶段都必须求解成功；超时、不可行或残差超过容差直接报错。
不把可行 incumbent 冒充 `C*`，也不静默改约束或输出“最优”账单。

## 日内执行与实际结算

`replay_day` 先完整求出午夜计划，再逐段调用 `observe(t)`。
每次只用当前实际供需替换剩余预测的首段，未来仍使用午夜预测。
固定所有剩余区间 `g`，执行两层滚动LP，只实施首段动作。

- 执行第一层：最小化剩余预计紧急购电费 `sum(5*p*e)`，已付计划费是常数；
- 执行第二层：在相同形式的局部费用预算内改善 `F2`；
- 始终无第三层、无日内普通购电调整；
- 每次局部0.1%让步不等于全年费用只增加0.1%；
- `State(soc, previous_power_kw)` 跨时段和跨日传递。

真实账单单独计算：`fees[:,0]=p*g`，`fees[:,1]=5*p*e_actual`。
二者求和为实际总费用，剩余购电不退款，没有上调/下调交易费用。
辅助目标、假设的储能价值不能抵扣账单。

当前无终端储能价值，因此可能在窗口末放空电池或不储存没有当日用途的光伏。
这是有限视野基线的限制；不在本分支偷偷增加终端奖励或每日回充条件。

## 去掉互斥的准确含义

不加互斥符合本次用户决定，但“费用最小必然使所有最优解互斥”并非无条件成立。

1. 无负载、无光伏、`g=0`、SOC=6000时，`c=d=10 kWh`满足松弛平衡，
   只消耗少量内部电量，费用与空闲解同为0。正电价也排除不了这个等价解。
2. 满电10800 kWh、上一段充电1100 kW、功率变化上限1000 kW时，
   当段必须至少保持净充电100 kW。无负载/PV的单段松弛LP可通过同时充放电
   消耗损失，取得最小购电量100/6 kWh；真实电池此时无法同时满足继续充电与满电边界。

第二项已作为真实求解回归测试，而非仅靠口头判断。
每次计划报告 `max_overlap_kwh`、`overlap_intervals`。
计划中的电池路径只作为松弛模型诊断，不能直接当作实际运行表。
实际首段若重叠超过1e-6 kWh，执行器抛出 `NonphysicalSolution`，
不净化轨迹后掩盖效率/功率变化、不恢复二元互斥、不输出虚假的成功结算。
若出现此错误，需要明确讨论功率变化偏好与可执行性的处理；本版不擅自松弛。

## 接入队友预测

```python
from experiments.problem2.linear_planning import Config, State, plan_day, replay_day

cfg = Config()
state = State(soc=6000, previous_power_kw=0)
# forecast: (144,2), kW; price: (144,), yuan/kWh
plan = plan_day(forecast, price, state, cfg)
# plan['grid'] is the committed 144-interval purchase vector, kWh.
# For a complete replay, call replay_day directly (it plans before observing).
plan, detail, state = replay_day(forecast, price, observe, state, cfg)
# observe(t) returns only interval t's actual [load, PV], kW.
# Carry returned state into the next day's replay_day.
```

预测来源及发布日期由调用方保证；优化模块不读取训练数据、不重训、不调预测参数。
实际数组必须通过逐段观测接口提供，午夜计划函数本身没有实际值参数。
`plan['metadata']`、`detail['execution_logs']`记录每层状态、目标、耗时和残差。

## 运行与验证

仓库根目录下：

```text
python -m unittest discover -s tests -p test_q2_linear_planning.py -v
python -m experiments.problem2.linear_planning.run --days 2 --out experiments/problem2/linear_planning/runs/new-smoke
```

可用 `--forecast-npz file.npz` 接入含 `forecast[days,144,2]` 的队友预测。
CLI从1月1日开始连续传递状态；默认只用已有周期预测做2天真实数据冒烟检查，
不是预测开发，也不是正式全年比较。`--days 365` 可扩展回放，但不是本次已完成事项。
正式比较必须另外统一所有候选的一月共同预热；不能将本CLI不同控制器自身产生的
一月状态直接当作与exp003同初始状态的公平比较。本次不交付正式 `result2.xlsx`。

输出 `daily.csv`（费用汇总）、`intervals.csv`（逐段执行及结算）、
`dispatch.npz`（完整数组）、`solver.json`（配置、数据哈希、求解记录）。
既有输出目录不覆盖；`runs/`忽略入库。可提交的验证摘要见 `validation.json`。

当前下一步：由预测负责人给出冻结预测与时间元数据，在统一预热和协议下回放，
记录费用、平稳性、重叠及失败情况，再决定是否增加备用/场景或终端处理。
