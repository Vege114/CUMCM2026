# exp007：RL 降低电池周转，但购电费用上升

## 1. 结论与成绩

**本轮固定预算 RL 未降低总费用，不能替代同预测基线作为最低费用方案。**

- 预声明主组 `regularized_42` 的 334 日真实总费为 **1,792.58 万元**，相对 exp004 无季节 **+431.42 万元（+31.69%）**；相对 exp006 正式树DP **+385.95 万元**。
- 电芯等效全循环为 **247.42 次**，相对同预测基线 **-50.43%**；非空方向反转 **1,129 次**，基线 **7,715 次**。这些描述运行强度，不代表已证实寿命收益。
- 六组训练预算和角色提前冻结。正式组保持 seed42；其余种子和纯费用对照全部展示，不按全年最低账单换主组。

主组相对同预测基线：计划费变化 **+304.02 万元**，紧急费变化 **+127.40 万元**，合计 **+431.42 万元**。

当前结论来自真实执行账单，奖励中的软罚与库存估值不计入题目费用。PPO 固定预算的训练诊断不能证明收敛或最优。

### 主组与同预测历史锚点

| 方案 | 角色 | 计划费/元 | 紧急费/元 | 总费/元 | 紧急/kWh |
|---|---|---|---|---|---|
| exp004 无季节 | 同预测基线 | 12,729,841.3665 | 881,746.4664 | 13,611,587.8329 | 160,912.1654 |
| exp006 正式·树DP | 正式策略 | 13,215,791.4493 | 850,466.0280 | 14,066,257.4773 | 197,559.7826 |
| exp006 旧执行·重新规划 | 消融 | 13,127,816.4567 | 406,380.2706 | 13,534,196.7273 | 74,976.2978 |
| RL 正式 · 42 | 正式策略 | 15,770,026.2460 | 2,155,729.3384 | 17,925,755.5844 | 488,218.1463 |
| RL 纯费用 · 42 | 预声明仅费用消融 | 15,565,443.6128 | 2,159,823.0264 | 17,725,266.6392 | 514,936.3779 |

## 2. 题目指标及信息边界

每天零点一次锁定全部 144 槽普通购电，日内不更改；电价与负载/光伏预测在规划时已知，实际供需仅在该槽执行时揭示。普通购电全部计费，实际账单为 **C=Σp(g+5e)**；没有售电收入、普通计划调整费或年末库存残值抵扣。

功率为 kW，十分钟电量为功率/6 kWh。SOC 使用电芯侧能量；充放电量使用交流母线侧能量，两者通过单向效率 sqrt(0.9) 连接。最大容量 12000 kWh，SOC 1200–10800 kWh，单向功率最多 5000 kW。主组没有额外硬爬坡上限。

### 费用、预测和电池指标

| 指标 | 定义 | 单位 | 范围 |
|---|---|---|---|
| MAE | Σ\|预测−实际\|/n | kW | 目标与总体分别核算 |
| RMSE | sqrt(Σ(预测−实际)²/n) | kW | 不能平均月度RMSE |
| WAPE | 100×Σ\|预测−实际\|/Σ\|实际\|；零分母为空 | % | 绝对差用百分点 |
| 题目费用 | Σp×计划购电+Σ5p×紧急购电 | 元 | 奖励罚项不计入 |
| 电芯EFC | (ηΣ充电+Σ放电/η)/(2×12000) | 次 | 运行强度代理 |
| 净功率变化 | ΔP=P_t−P_(t−1)，连续334日含跨午夜 | kW/相邻10分钟 | 预热边界另报，不补首点零 |
| 非空方向反转 | 去除闲置后，相邻非空动作方向相反 | 次 | 与直接相邻槽反转分开 |

互斥、SOC 和单向功率限额由有符号动作及物理投影保证；轻惩罚是费用之外的次要偏好。三项软罚为 0.002 元/交流侧kWh 吞吐、0.05 元/非空反转、0.0002 元/kW 相邻功率变化。纯费用消融将三项都置零，保留相同物理、动作、2 kWh 死区及终值规则。不能把这些未标定偏好称为实际电池折旧。

## 3. 数据与时间验证

### 输入校验值

| 文件 | SHA-256 |
|---|---|
| 附件1.csv | 99bae60824c8e80b5b74b180b3f0b76f2649059c437f2a2878d0a7f93e8c598d |
| 附件2_小区负载.csv | a2e0e3e0e35a3551f67a747aeccd308fa745c8a8b6d20bba5d57f05dd9817c97 |
| 附件2_光伏发电实际功率.csv | 8889416ff7231b832487a0d1c8629f10cf321a1d7cc5453043b003f864397fce |

正式评价为 **2025-02-01 至 12-31，334 日、每组 48,096 槽**。附件2按区间终点解释，00:10 是当天第一段，24:00 是当天最后一段。初始状态承接真实一月预热，2 月 1 日 SOC 1421.7991105135516 kWh、上一槽净功率 172.76 kW；各组此后延续自己的实际状态。

首次训练只用 1 月 9–31 日的因果周期预测回退，后续每 14 日只用已完成的最近至多90日继续训练。当天144槽计划锁定前不访问当天实况奖励。预报、训练 cutoff、checkpoint 与执行档案的哈希和因果核验单独保留。

六组统一复用 exp004 `no_season/seed42` 冻结预测，SHA-256 `6f7bcf27e420e2f05251820a5c1d8d1f87cd6234287e23505b33a51ae6d757b9`；没有重训预测网络。RL 随机种子与预测种子分别记录。旧实验共享早停、回顾性评价和含未来探索的边界沿历史记录说明，不能因本轮逐槽因果就抹去历史研究的先验接触。

## 4. 逐步技术讲解

| 网络层 | 维度 | 变换 | 用途 |
|---|---|---|---|
| 观测输入 | 29.0000 | 固定尺度 | 午夜预测与计划状态 |
| 共享层 1 | 64.0000 | tanh |  |
| 共享层 2 | 64.0000 | tanh |  |
| 策略头 | 45.0000 | 线性 logits → softmax | 训练抽样；正式 argmax |
| 价值头 | 1.0000 | 线性 | 估计训练日回报 |

### 1. 一天的 144 个决策都在午夜完成

这次实验复用 exp004 无季节预测、预测种子 42，每天零点一次读取未来 144 个十分钟槽的负载和光伏功率。PPO 学的是如何据此制定全天普通购电和电池意图动作。所有普通购电量锁定后，才按时间顺序读取当天实际供需并模拟执行。日内不会重新调用策略改变普通计划。不同 RL 种子改变策略训练，不改变预测数组。

规划环境只包含预测、已知固定电价、历史误差支持和计划 SOC，没有当天实际数组。策略逐槽前推的是计划状态；当天实况奖励只能在完整计划生成之后计算。这样避免策略在构造后续购电时借用尚未发生的实际供需。执行器使用当前槽实际值进行裁剪，是在线执行；PPO 本身是午夜开环规划。



### 2. 历史误差与冷启动

2 月起复用 exp006 的条件误差树接口。树只用零点前已完整揭晓的历史日残差，基于时刻、预测净需求、预测光伏和电价等特征，为各槽输出九个等权边际支持点。九列不代表九条具有联合天气相关性的日路径。净需求预测为 `(预测负载−预测光伏)/6`，从 kW 变为每十分钟 kWh。

1 月没有冻结的 exp004 预测，不能伪造其历史网络输出。首次训练使用 1 月 9–31 日共 23 个完整日的因果周期预测：负载取昨日与前周同槽，光伏取昨日同槽；每个历史发布日的误差支持也只能由该发布日之前的实际数据建立。首两天的正式历史误差支持同样遵守既有回退机制。冷启动来源与正式冻结预测存在分布差异，是需要保留的局限。

之后每 14 日继续训练一次，只使用该零点之前最近至多 90 个完整日。已经评分的过去日可进入后续训练，后续日不能反过来影响过去策略。这是回顾性的逐时序评价；历史实验结果已经被看到，不能将它称作从未接触过的独立测试集。



### 3. 29 维观测包含什么

每个规划槽的输入是 29 个实数：时刻的正弦和余弦、日内进度、归一化计划 SOC、上次非空意图方向、上一意图功率；本槽预测净需求、预测光伏、净需求支持的 80% 分位数、支持标准差；本槽电价、剩余日平均净需求、剩余日平均电价；未来八个两小时块各自的平均预测净需求和电价，共 16 项。临近日末时，超出日末的块按实现边界使用最后可用槽，不读取下一日未来实况。

SOC 以 1200–10800 kWh 归一化，功率除以 5000，净需求及其分位数除以 2000，光伏功率除以 12000，支持标准差除以 1000，电价除以当日固定最高电价。这些固定尺度不是用全年测试实况拟合的归一化器。该观测将全天预测压缩为当前和未来块均值，没有完整表示时间序列和真实状态不确定性，不能声称已满足完全可观测 Markov 状态。



### 4. 45 类动作怎样生成购电计划

每个动作编号分成电池档位和购电裕度：`battery_id=action//5`，`margin_id=action%5`。九档电池比例为 `−1, −0.75, −0.5, −0.25, 0, 0.25, 0.5, 0.75, 1`，乘以单槽上限 `5000/6 kWh`。正值充电，负值放电。充电按 `(10800−计划SOC)/η` 裁剪，放电按 `(计划SOC−1200)η` 裁剪；不足 2 kWh 的动作置零。因动作有符号，计划不能同时充放电。

设 `q80` 为九点支持的经验 80% 分位数，`σ=max(50,支持标准差)`，五档裕度为 `−1,−0.5,0,0.5,1`。普通购电为 `g=max(0,q80+c−d+margin×σ)`。80% 分位数来自普通费率与五倍紧急费率的单槽权衡；PPO 可以通过裕度改变该中心，也选择电池意图。有限档位和这种参数化限制了可搜索计划的范围，不能保证包含连续最优解。



### 5. 网络、抽样和 PPO 更新

29 维输入经过两个 64 单元的 tanh 全连接层，形成共享表示。一条输出头给出 45 个动作的 logits，softmax 将它们转为分类策略 `πθ(a|o)`；另一条线性输出头给出标量价值 `Vθ(o)`。训练从分类策略抽样，正式午夜规划选最大 logit 的动作，不采样，也不对多个种子投票。

每次收集 32 个完整日回合，即 `144×32=4608` 个转移。历史日按当前可用训练池有放回抽样；初始 SOC 在允许范围随机化，并在每批包含共同 2 月初状态。训练日的初始方向和上一功率使用实现规定的训练默认边界；正式年度评价则连续传递真实边界。因此随机日训练并不等价于沿整年连续状态分布训练。

完整日结束后才计算真实执行奖励。广义优势估计按反向递推：`δt=rt+γV(o[t+1])−V(ot)`，`At=δt+γλA[t+1]`；完整回合末端的价值 bootstrap 为零。正式协议使用 `γ=1`、`λ=0.95`。价值目标为 `Rt=At+V(ot)`。优势只在本次已提供的训练 rollout 内标准化，不使用验证或未来日。

策略概率比为 `ρt=πθ(at|ot)/πold(at|ot)`。PPO 策略损失是 `−mean(min(ρtAt,clip(ρt,0.8,1.2)At))`。总优化损失再加 `0.5×mean((Vθ−R)²)`，减去 `0.01×策略熵`。Adam 学习率为 0.0003，每批做四轮、每小批 512 条；全局梯度范数裁剪到 0.5。近似 KL、裁剪比例、策略熵和价值损失作为训练诊断保存。每 14 日历史训练池会变化，从冷启动 23 日逐步变为最近至多90日，所以跨截止点的训练回合均值不能直接解释为收敛或独立验证改善。损失下降、有限 KL 或梯度通过核验都不构成策略已收敛或全局最优的证明。

首次固定 218 次更新；此后 23 个截止点各继续 44 次，总计每种子 1230 次、5,667,840 个训练转移。每个截止点使用固定预算后的策略，不通过未来日账单选最优 checkpoint、停止轮数或超参数。检查点保存网络、优化器和随机状态，签名还绑定数据、源码、协议和历史截止边界。



### 6. 奖励里电费是主体

每槽真实购电费为 `Ct=pt(gt+5et)`，普通购电全部计费、弃电无收入。轻惩罚组奖励为：

`rt=−[Ct+0.002(ct+dt)+0.05×非空方向反转+0.0002|Pt−P[t−1]|]/1000`。

电量单位为 kWh，`Pt=6(ct−dt)` 为母线净功率 kW。非空反转忽略中间空闲槽；它与新报告要求的相邻槽直接反转是两种指标。纯费用消融只把三项电池软罚设零，保留相同动作集合、真实约束、执行器和 2 kWh 死区。因此它检验的是这组三项轻偏好的组合，不能分别归因每个惩罚。

完整训练日末另加谷价库存变化估值 `min(p)/η×(实际末SOC−初SOC)/1000`，以减少回合结束时无条件放空的偏差。该估值不从真实全年账单抵扣；最终评价日关闭的是奖励核算中的终值项。当前 actor 没有年末日期特征，因此已经学到的库存偏好不会因关闭核算项而自动消失。它仍是日回合近似，不是严格全年价值函数。三个软罚没有被标定为化学电池真实折旧费。



### 7. 执行投影与连续状态

给定锁定购电 `g`，本槽余缺为 `B=g+(实际光伏−实际负载)/6`。余电时只允许充电，取意图充电、可用余电、功率限额和剩余容量的最小值；缺电时只允许放电，取意图放电、缺口、功率限额和可用 SOC 的最小值。2 kWh 死区处理后，剩余缺口紧急补购，剩余余电弃置。没有紧急购电主动充电，也没有同时充放电。

电芯状态满足 `SOC_next=SOC+ηc−d/η`，`η=sqrt(0.9)`；容量 12000 kWh，SOC 1200–10800 kWh，单向功率 5000 kW。2 月 1 日共同初始 SOC 为 1421.7991105135516 kWh、上一槽净功率 172.76 kW，随后每个策略延续自己的真实末状态、最近非空方向和末槽功率，不每日复位。没有额外硬爬坡约束，软罚不足以保证功率变化不超过 1000 kW/10min。

实际裁剪会偏离计划 SOC 和意图动作。比如计划放电而实际出现余电时，意图充电为零可能导致弃电；反之可能额外紧急补购。PPO 的奖励通过真实执行计算，能学习这些后果，但它仍只观察午夜计划状态，不能保证消除这种规划与执行差异。全年费用必须用实际执行档案重新结算，不能由训练奖励代替。



### 8. 结论的适用边界

主组固定为 `regularized_42`，另外两个轻惩罚种子评价训练稳定性，三个纯费用种子是配对消融。均值和样本标准差仅概括这三个训练种子；不同种子高度共享同一年供需，不能据此宣称跨年度泛化或统计显著。所有方案使用同一预测，费用变化来自本次规划及执行轨迹。

与 exp004 无季节、exp006 正式及其原角色消融的比较优先保持原始单位和相同计费/物理边界。exp005 有硬爬坡及不同紧急充电规则，只作描述性历史；exp001 原登记及含未来的 exp004 全年探索不进入直接排名。若 RL 更贵或电池更频繁动作，保留负面结论，不以耗时、奖励或某个有利种子替代主目标。


### 规划、隐藏奖励与真实执行的信息流

**午夜规划**：冻结144×2预测 → 历史误差九点支持 → 29维观测 → PPO逐槽选45类动作 → 锁定全天购电与意图

**训练更新**：只抽已完成历史日 → 32个完整日rollout → 计划完成后执行并计奖励 → GAE优势估计 → 四轮PPO更新

**年度执行**：固定午夜计划 → 仅本槽实际供需 → 意图动作按物理裁剪 → 紧急补购或弃电 → 真实SOC跨日

## 5. 实验设置

### 冻结算法、训练与物理配置

| 类别 | 参数 | 值 |
|---|---|---|
| algorithm | name | "categorical_PPO" |
| algorithm | observations | 29 |
| algorithm | actions | 45 |
| algorithm | battery_levels | 9 |
| algorithm | purchase_margin_levels | 5 |
| algorithm | hidden_layers | [64, 64] |
| algorithm | activation | "tanh" |
| algorithm | learning_rate | 0.0003 |
| algorithm | clip_ratio | 0.2 |
| algorithm | entropy_coefficient | 0.01 |
| algorithm | value_coefficient | 0.5 |
| algorithm | max_gradient_norm | 0.5 |
| algorithm | gamma | 1.0 |
| algorithm | gae_lambda | 0.95 |
| algorithm | epochs_per_update | 4 |
| algorithm | minibatch_size | 512 |
| algorithm | parallel_envs | 32 |
| training | first_cutoff_day | 31 |
| training | initial_iterations | 218 |
| training | continue_every_days | 14 |
| training | online_iterations_each | 44 |
| training | online_updates | 23 |
| training | total_iterations_per_run | 1230 |
| training | transitions_per_run | 5667840 |
| training | history_days_max | 90 |
| training | january_first_training_day | 8 |
| training | january_days | 23 |
| training | january_source | "past_only_periodic_baseline_not_exp004" |
| training | history_cutoff_exclusive | true |
| training | actual_reward_only_after_complete_day_plan | true |
| training | sample_complete_day_paths | true |
| training | random_training_soc | [1200, 10800] |
| training | hyperparameter_selection | "fixed_before_formal_evaluation; env count selected by pilot speed only" |
| training | policy_selection | "fixed final update at each causal cutoff; do not select using future evaluation" |
| training | terminal_training_shaping | "min_tariff/eta * (final_actual_soc-initial_soc); no reported credit; final evaluation day disabled" |
| training | pilot_policy_reuse | false |
| battery | nominal_capacity_kwh | 12000 |
| battery | min_soc_kwh | 1200 |
| battery | max_soc_kwh | 10800 |
| battery | max_power_kw | 5000 |
| battery | charge_efficiency | 0.9486832980505138 |
| battery | discharge_efficiency | 0.9486832980505138 |
| battery | simultaneous_charge_discharge | "forbidden_by_signed_action_and_projection" |
| battery | emergency_charging | false |
| battery | hard_ramp_limit_kw | null |
| battery | aging_proxy_only | true |

### 六组角色与训练种子

| 方案 | 角色 | RL种子 | 评价日 | 训练转移 | 任务墙钟/s |
|---|---|---|---|---|---|
| RL 正式 · 42 | 正式策略 | 42.0000 | 334.0000 | 5,667,840.0000 | 88.3645 |
| RL 轻惩罚 · 2026 | 训练种子稳定性 | 2,026.0000 | 334.0000 | 5,667,840.0000 | 90.4257 |
| RL 轻惩罚 · 3407 | 训练种子稳定性 | 3,407.0000 | 334.0000 | 5,667,840.0000 | 89.4875 |
| RL 纯费用 · 42 | 预声明仅费用消融 | 42.0000 | 334.0000 | 5,667,840.0000 | 98.1310 |
| RL 纯费用 · 2026 | 预声明仅费用消融 | 2,026.0000 | 334.0000 | 5,667,840.0000 | 102.3961 |
| RL 纯费用 · 3407 | 预声明仅费用消融 | 3,407.0000 | 334.0000 | 5,667,840.0000 | 106.0030 |

轻惩罚与纯费用各有 RL seed42、2026、3407，共六组。正式主组提前固定为 `regularized_42`；其他轻惩罚种子检验训练稳定性，纯费用三种子是配对消融。每种子固定 1230 次更新、5,667,840 个转移，无未来费用选模、无种子集成。

网络和优化在 CPU 单线程运行，32 个批量环境同步构造日计划。任务墙钟包含准备、训练、预测规划、执行与存储，阶段累计有包含关系，不能与墙钟相加。报告和工作簿导出另计。

## 6. 结果及失败案例

### 全部六组真实费用与电池结果

| 方案 | 计划费/元 | 紧急费/元 | 总费/元 | 紧急/kWh | 电芯EFC | 非空反转 | 同时槽 | 期末SOC/kWh |
|---|---|---|---|---|---|---|---|---|
| RL 正式 · 42 | 15,770,026.2460 | 2,155,729.3384 | 17,925,755.5844 | 488,218.1463 | 247.4181 | 1,129.0000 | 0.0000 | 1,200.0000 |
| RL 轻惩罚 · 2026 | 15,840,429.9813 | 1,717,856.5732 | 17,558,286.5544 | 398,841.2959 | 278.3182 | 1,296.0000 | 0.0000 | 1,200.0000 |
| RL 轻惩罚 · 3407 | 15,650,223.4487 | 2,032,331.8045 | 17,682,555.2532 | 468,151.7976 | 276.7336 | 1,137.0000 | 0.0000 | 1,200.0000 |
| RL 纯费用 · 42 | 15,565,443.6128 | 2,159,823.0264 | 17,725,266.6392 | 514,936.3779 | 287.4799 | 1,090.0000 | 0.0000 | 1,200.0000 |
| RL 纯费用 · 2026 | 16,017,082.9970 | 1,610,024.2760 | 17,627,107.2730 | 384,312.5505 | 259.9328 | 1,234.0000 | 0.0000 | 1,200.0000 |
| RL 纯费用 · 3407 | 15,685,812.0795 | 1,851,515.5514 | 17,537,327.6309 | 434,600.0974 | 286.0449 | 1,229.0000 | 0.0000 | 1,200.0000 |

六组真实费用都高于 exp006 正式策略与其旧执行重规划对照，固定预算下未达到降本目标。轻惩罚三种子总费均值 **1,772.22 万元**，纯费用三种子均值 **1,762.99 万元**。

同为RL seed42时，轻惩罚比纯费用多 **200,488.95 元（+1.13%）**。EFC 从 **287.48** 降到 **247.42**，但非空反转从 **1,090** 增到 **1,129**，直接相邻反转从 **246** 增到 **433**。轻惩罚没有让每项电池指标都改善；只能解释为本次组合和训练轨迹的结果。

### 三种子均值与样本标准差

| 组别 | 指标 | 样本数 | 均值 | 样本标准差 |
|---|---|---|---|---|
| 轻惩罚 | 总费用/元 | 3.0000 | 17,722,199.1307 | 186,914.6858 |
| 轻惩罚 | 紧急费/元 | 3.0000 | 1,968,639.2387 | 225,777.9813 |
| 轻惩罚 | 紧急购电/kWh | 3.0000 | 451,737.0799 | 46,894.9631 |
| 轻惩罚 | 交流吞吐/kWh | 3.0000 | 6,428,657.3791 | 418,198.5697 |
| 轻惩罚 | 电芯EFC/次 | 3.0000 | 267.4900 | 17.4008 |
| 轻惩罚 | direct_adjacent_reversals | 3.0000 | 442.3333 | 79.4124 |
| 轻惩罚 | nonidle_direction_reversals | 3.0000 | 1,187.3333 | 94.1931 |
| 轻惩罚 | 功率总变差/kW | 3.0000 | 7,376,750.9183 | 323,046.8249 |
| 纯费用 | 总费用/元 | 3.0000 | 17,629,900.5144 | 94,000.6349 |
| 纯费用 | 紧急费/元 | 3.0000 | 1,873,787.6179 | 275,575.2169 |
| 纯费用 | 紧急购电/kWh | 3.0000 | 444,616.3419 | 65,885.4304 |
| 纯费用 | 交流吞吐/kWh | 3.0000 | 6,676,903.3444 | 372,676.8920 |
| 纯费用 | 电芯EFC/次 | 3.0000 | 277.8192 | 15.5067 |
| 纯费用 | direct_adjacent_reversals | 3.0000 | 377.0000 | 124.5913 |
| 纯费用 | nonidle_direction_reversals | 3.0000 | 1,184.3333 | 81.7333 |
| 纯费用 | 功率总变差/kW | 3.0000 | 7,690,065.5772 | 299,149.9863 |

正式主组最贵日为 **2025-12-02**，日费 **94,312.33 元**，紧急购电 **5,410.01 kWh**。这里选择最不利日期说明局限，随机日电池展示则独立按固定种子抽样。

随机种子 **20260912** 从完整评价日不放回抽四日，所有策略用相同日期与纵轴：2025-04-15、2025-06-07、2025-12-06、2025-12-25。图中充电为正、放电为负，十分钟电量乘6得到kW。全年视图包含全部原始点；可选单日或日期范围放大，并用滑块查看每个原始槽。图中±5000 kW为功率边界，未混入1月预热。

正式期相邻功率差包含跨午夜；首点没有差分，预热末点到正式首点的跳变单列。大跳变定义为 |ΔP|>1000 kW/10min，仅为描述阈值。幅值受限、SOC平滑和功率变化受限是不同事实，不能互相替代。

### 完整时间轴的功率波动指标

| 方案 | 平均|ΔP|/kW | 变化RMS/kW | P95|ΔP|/kW | 最大|ΔP|/kW | 总变差/kW | 相邻直接反转 | 贴限/% | 大跳变/% | 预热跳变/kW |
|---|---|---|---|---|---|---|---|---|---|
| exp004 无季节 | 695.0917 | 1,636.9222 | 4,750.5609 | 9,704.3668 | 33,430,432.9182 | 6,796.0000 | 8.4227 | 14.3778 | 4,827.2400 |
| exp006 正式·树DP | 633.1436 | 1,591.6255 | 4,714.7898 | 9,013.7957 | 30,451,039.1435 | 730.0000 | 0.0000 | 14.6585 | 4,749.0146 |
| exp006 旧执行·重新规划 | 679.2005 | 1,582.4537 | 4,522.7537 | 9,255.1867 | 32,666,146.3283 | 5,731.0000 | 4.9401 | 14.5545 | 4,827.2400 |
| RL 正式 · 42 | 149.3877 | 569.6472 | 553.1292 | 7,316.9777 | 7,184,801.2270 | 433.0000 | 1.0687 | 3.6511 | 1,077.2400 |
| RL 轻惩罚 · 2026 | 161.1336 | 568.0854 | 653.6101 | 7,093.0867 | 7,749,719.8760 | 526.0000 | 0.2100 | 4.1065 | -502.5941 |
| RL 轻惩罚 · 3407 | 149.6150 | 516.8641 | 609.0638 | 7,675.8719 | 7,195,731.6519 | 368.0000 | 0.2162 | 3.7717 | 1,077.2400 |
| RL 纯费用 · 42 | 154.9423 | 539.4588 | 726.9887 | 8,230.2036 | 7,451,951.4634 | 246.0000 | 0.1289 | 4.1875 | -502.5941 |
| RL 纯费用 · 2026 | 157.8626 | 594.8797 | 627.9174 | 7,410.7251 | 7,592,404.0010 | 494.0000 | 0.1954 | 4.0087 | -502.5941 |
| RL 纯费用 · 3407 | 166.8748 | 590.2847 | 708.1436 | 7,430.2138 | 8,025,841.2671 | 391.0000 | 0.2079 | 4.2125 | 1,077.2400 |

主组平均绝对功率变化为 **149.39 kW/10min**，RMS **569.65**，但最大跳变仍达 **7,316.98 kW/10min**。相邻直接反转 **433 次**，大于1000 kW的变化占 **3.65%**。这些结果不构成硬爬坡保证。

### 电池吞吐与非空方向反转

| 方案 | 充电/kWh | 放电/kWh | 电芯EFC | 非空反转 | 同时槽 | 初SOC/kWh | 末SOC/kWh |
|---|---|---|---|---|---|---|---|
| exp001 同口径重算 | 6,276,030.0368 | 5,648,637.4502 | 496.1730 | — | — | — | 1,200.0000 |
| exp002 正式风险 | 6,316,145.3262 | 5,684,741.2107 | 499.3444 | 8,881.0000 | 0.0000 | 1,421.7991 | 1,200.0000 |
| exp003 正式 | 6,341,190.3894 | 5,707,232.2343 | 501.3222 | 8,809.0000 | 0.0000 | 1,421.7991 | 1,252.2127 |
| exp004 无季节 | 6,313,399.0876 | 5,682,140.7682 | 499.1216 | 7,715.0000 | 0.0000 | 1,421.7991 | 1,335.7963 |
| exp004 历史季节 | 6,317,717.4916 | 5,685,938.6037 | 499.4591 | 7,619.0000 | 0.0000 | 1,421.7991 | 1,429.3240 |
| exp006 正式·树DP | 6,042,575.4367 | 5,438,271.6032 | 477.7055 | 2,729.0000 | 0.0000 | 1,421.7991 | 1,470.5929 |
| exp006 旧执行·重新规划 | 6,190,052.9047 | 5,570,850.3089 | 489.3580 | 6,755.0000 | 0.0000 | 1,421.7991 | 1,629.7771 |
| exp005 LP·β=0.1 | 6,059,525.5222 | 5,453,783.3487 | 479.0568 | 3,859.0000 | 0.0000 | 1,421.7991 | 1,200.0404 |
| exp005 LP·β=0.01 | 6,116,754.5320 | 5,505,289.4576 | 483.5811 | 3,779.0000 | 0.0000 | 1,421.7991 | 1,200.0404 |
| RL 正式 · 42 | 3,129,501.9526 | 2,816,762.1745 | 247.4181 | 1,129.0000 | 0.0000 | 1,421.7991 | 1,200.0000 |
| RL 纯费用 · 42 | 3,636,248.3018 | 3,272,833.8887 | 287.4799 | 1,090.0000 | 0.0000 | 1,421.7991 | 1,200.0000 |

训练曲线显示固定预算内的优化行为。价值损失随不同历史池、随机初始状态和奖励分布可改变，不能只用其绝对大小判断优劣；策略熵下降也可能伴随过早确定化。每14日历史训练池变化，跨截止点均值变化不能称为收敛或独立验证改善。末日关闭终值只影响奖励核算，actor没有年末特征，已学到的库存偏好不会自动消失。PPO 没有全局最优证书，本轮没有凭全年费用调整训练预算。

### 题目四个指定日 · 正式组日汇总

| 日期 | 计划/kWh | 计划费/元 | 紧急费/元 | 总费/元 | 日初SOC/kWh | 日末SOC/kWh |
|---|---|---|---|---|---|---|
| 2025-03-20 | 62,293.9273 | 49,610.1027 | 13,974.7323 | 63,584.8350 | 10,800.0000 | 9,958.1256 |
| 2025-06-21 | 37,059.4321 | 26,594.4632 | 20.2678 | 26,614.7310 | 3,643.9907 | 3,532.2468 |
| 2025-09-23 | 68,537.8771 | 47,117.8591 | 11,615.5570 | 58,733.4161 | 1,200.0000 | 1,200.0000 |
| 2025-12-21 | 90,173.7733 | 65,093.8892 | 2,403.8031 | 67,497.6923 | 3,222.6319 | 1,837.8075 |

### 表1 · 六个指定十分钟区间

| 日期 | 区间 | 计划/kWh |
|---|---|---|
| 2025-03-20 | 10:00-10:10 | 0.0000 |
| 2025-03-20 | 12:00-12:10 | 0.0000 |
| 2025-03-20 | 14:00-14:10 | 0.0000 |
| 2025-03-20 | 16:00-16:10 | 412.4444 |
| 2025-03-20 | 18:00-18:10 | 669.6574 |
| 2025-03-20 | 20:00-20:10 | 688.9489 |
| 2025-06-21 | 10:00-10:10 | 0.0000 |
| 2025-06-21 | 12:00-12:10 | 0.0000 |
| 2025-06-21 | 14:00-14:10 | 0.0000 |
| 2025-06-21 | 16:00-16:10 | 153.4329 |
| 2025-06-21 | 18:00-18:10 | 470.8656 |
| 2025-06-21 | 20:00-20:10 | 667.8800 |
| 2025-09-23 | 10:00-10:10 | 0.0000 |
| 2025-09-23 | 12:00-12:10 | 0.0000 |
| 2025-09-23 | 14:00-14:10 | 0.0000 |
| 2025-09-23 | 16:00-16:10 | 519.1537 |
| 2025-09-23 | 18:00-18:10 | 747.1781 |
| 2025-09-23 | 20:00-20:10 | 0.0000 |
| 2025-12-21 | 10:00-10:10 | 615.7816 |
| 2025-12-21 | 12:00-12:10 | 277.6734 |
| 2025-12-21 | 14:00-14:10 | 574.3752 |
| 2025-12-21 | 16:00-16:10 | 773.6368 |
| 2025-12-21 | 18:00-18:10 | 577.2066 |
| 2025-12-21 | 20:00-20:10 | 0.0000 |

### 表2 · 六个四小时段电池电量

| 日期 | 四小时段 | 充电/kWh | 放电/kWh |
|---|---|---|---|
| 2025-03-20 | 0:00-4:00 | 0.0000 | 0.0000 |
| 2025-03-20 | 4:00-8:00 | 0.0000 | 0.0000 |
| 2025-03-20 | 8:00-12:00 | 0.0000 | 0.0000 |
| 2025-03-20 | 12:00-16:00 | 0.0000 | 0.0000 |
| 2025-03-20 | 16:00-20:00 | 0.0000 | 0.0000 |
| 2025-03-20 | 20:00-24:00 | 0.0000 | 798.6722 |
| 2025-06-21 | 0:00-4:00 | 4,557.9996 | 383.9458 |
| 2025-06-21 | 4:00-8:00 | 161.6784 | 5,255.2544 |
| 2025-06-21 | 8:00-12:00 | 4,913.2033 | 0.0000 |
| 2025-06-21 | 12:00-16:00 | 4,175.9928 | 0.0000 |
| 2025-06-21 | 16:00-20:00 | 0.0000 | 0.0000 |
| 2025-06-21 | 20:00-24:00 | 0.0000 | 6,894.7961 |
| 2025-09-23 | 0:00-4:00 | 4,590.1613 | 0.0000 |
| 2025-09-23 | 4:00-8:00 | 4,417.9063 | 0.0000 |
| 2025-09-23 | 8:00-12:00 | 119.2885 | 0.0000 |
| 2025-09-23 | 12:00-16:00 | 0.0000 | 0.0000 |
| 2025-09-23 | 16:00-20:00 | 0.0000 | 5,494.6920 |
| 2025-09-23 | 20:00-24:00 | 0.0000 | 2,719.9285 |
| 2025-12-21 | 0:00-4:00 | 4,334.2366 | 0.0000 |
| 2025-12-21 | 4:00-8:00 | 2,780.0878 | 2,819.9665 |
| 2025-12-21 | 8:00-12:00 | 2,330.5393 | 2,775.5836 |
| 2025-12-21 | 12:00-16:00 | 3,842.7169 | 0.0000 |
| 2025-12-21 | 16:00-20:00 | 0.0000 | 3,127.8861 |
| 2025-12-21 | 20:00-24:00 | 0.0000 | 4,549.1460 |

### 表3 · 指定日全部紧急购电区间

| 日期 | 连续区间 | 紧急/kWh |
|---|---|---|
| 2025-03-20 | 00:10-00:40 | 73.5921 |
| 2025-03-20 | 00:50-02:10 | 384.4349 |
| 2025-03-20 | 02:20-04:10 | 355.8941 |
| 2025-03-20 | 04:20-05:30 | 225.3313 |
| 2025-03-20 | 05:40-06:10 | 71.2661 |
| 2025-03-20 | 06:20-07:00 | 122.2091 |
| 2025-03-20 | 08:00-08:30 | 69.0892 |
| 2025-03-20 | 09:00-10:00 | 116.9658 |
| 2025-03-20 | 14:10-14:30 | 32.5507 |
| 2025-03-20 | 14:50-17:40 | 1,147.3140 |
| 2025-03-20 | 18:00-19:20 | 301.7433 |
| 2025-03-20 | 19:30-22:30 | 537.2630 |
| 2025-03-20 | 22:40-23:10 | 35.4435 |
| 2025-03-20 | 23:20-23:50 | 57.1864 |
| 2025-06-21 | 04:00-04:10 | 8.7379 |
| 2025-06-21 | 05:00-05:10 | 0.4800 |
| 2025-09-23 | 08:20-08:50 | 171.1232 |
| 2025-09-23 | 09:10-09:50 | 124.9249 |
| 2025-09-23 | 10:00-10:10 | 3.3890 |
| 2025-09-23 | 14:20-15:20 | 704.6899 |
| 2025-09-23 | 15:50-16:50 | 349.8504 |
| 2025-09-23 | 17:00-17:50 | 202.1683 |
| 2025-09-23 | 18:00-18:50 | 163.6073 |
| 2025-09-23 | 20:10-20:20 | 10.2344 |
| 2025-09-23 | 20:30-20:40 | 465.7332 |
| 2025-09-23 | 20:50-21:00 | 37.9974 |
| 2025-09-23 | 21:20-21:30 | 16.4217 |
| 2025-09-23 | 22:20-22:30 | 5.8906 |
| 2025-12-21 | 06:30-06:40 | 26.2802 |
| 2025-12-21 | 07:50-08:00 | 18.8978 |
| 2025-12-21 | 15:20-15:30 | 69.4718 |
| 2025-12-21 | 15:50-16:30 | 130.2555 |
| 2025-12-21 | 16:40-18:00 | 255.0457 |
| 2025-12-21 | 21:30-21:40 | 4.0537 |
| 2025-12-21 | 22:50-23:10 | 19.7344 |
| 2025-12-21 | 23:30-23:50 | 35.8877 |

[原题全年工作簿](result2.xlsx)与[指定四日完整表格](specified_dates.md)按正式主组输出。工作簿保留 334 日三张题目表，涵盖144槽普通购电、六个四小时段充放电与日初日末SOC、全部紧急购电区间。原始表格及读回证据同包保存；四小时聚合充放电均为正不意味着同一个十分钟槽同时充放电。另保留[纯费用seed42工作簿](cost_only_42/result2.xlsx)及其[指定日表](cost_only_42/specified_dates.md)，仍为消融角色。

![monthly-costs](figures/monthly-costs.png)

![seed-costs](figures/seed-costs.png)

![battery-comparison](figures/battery-comparison.png)

![power-variation](figures/power-variation.png)

![learning-curves](figures/learning-curves.png)

![daily-cost-differences](figures/daily-cost-differences.png)

![failure-case](figures/failure-case.png)

![battery-random-days](figures/battery-random-days.png)

![battery-full-evaluation](figures/battery-full-evaluation.png)

![exp004__no_season-full](figures/exp004__no_season-full.png)

![exp004__no_season-random](figures/exp004__no_season-random.png)

![exp006__primary-full](figures/exp006__primary-full.png)

![exp006__primary-random](figures/exp006__primary-random.png)

![exp006__greedy_execution-full](figures/exp006__greedy_execution-full.png)

![exp006__greedy_execution-random](figures/exp006__greedy_execution-random.png)

![exp007__regularized_42-full](figures/exp007__regularized_42-full.png)

![exp007__regularized_42-random](figures/exp007__regularized_42-random.png)

![exp007__regularized_2026-full](figures/exp007__regularized_2026-full.png)

![exp007__regularized_2026-random](figures/exp007__regularized_2026-random.png)

![exp007__regularized_3407-full](figures/exp007__regularized_3407-full.png)

![exp007__regularized_3407-random](figures/exp007__regularized_3407-random.png)

![exp007__cost_only_42-full](figures/exp007__cost_only_42-full.png)

![exp007__cost_only_42-random](figures/exp007__cost_only_42-random.png)

![exp007__cost_only_2026-full](figures/exp007__cost_only_2026-full.png)

![exp007__cost_only_2026-random](figures/exp007__cost_only_2026-random.png)

![exp007__cost_only_3407-full](figures/exp007__cost_only_3407-full.png)

![exp007__cost_only_3407-random](figures/exp007__cost_only_3407-random.png)

## 7. 历次指标和技术路线对比

### 协议不同或含未来 · 保留原始成绩

| 策略 | 总费/元 | 日数 | 原角色 | 可比性说明 |
|---|---|---|---|---|
| exp004 全年探索 | 13,572,926.4786 | 334.0000 | 含未来探索 | 含未来信息，仅探索 |
| exp001 原登记（不可比） | 16,258,308.2174 | 334.0000 | 原登记不可比 | 原协议不可直接比较 |
| exp005 LP·β=0.1 | 14,717,882.1655 | 334.0000 | 当轮正式策略 | 仅描述：1000 kW硬爬坡、LP松弛及允许紧急充电；不排名 |
| exp005 LP·β=0.01 | 14,740,993.8739 | 334.0000 | 历史权重对照 | 仅描述：1000 kW硬爬坡、LP松弛及允许紧急充电；不排名 |

exp001 原登记与同物理重算分别保留；exp001/2 早期共享四目标早停边界照实列示。exp004 全年探索含未来实际信息，不能作为可用策略排名。exp004 无季节是本轮同预测锚点，历史季节保留当轮正式角色。

exp005 的供需与电价数值、时间轴已由证据代理单独核对，但其 **1000 kW/10min 硬爬坡** 与 **允许紧急购电服务充电** 的执行协议不同，费用仅作描述性对照。exp006 正式树DP、旧执行重规划、固定购电旧执行保持原角色；不能因为其中消融费用较低而改写它的正式策略。

### 费用比较原值、绝对差和相对变化

| 历史策略 | 历史/元 | RL/元 | 差值/元 | 相对变化/% | 可比 |
|---|---|---|---|---|---|
| exp001 同口径重算 | 15,842,654.1907 | 17,925,755.5844 | 2,083,101.3937 | 13.1487 | 是 |
| exp002 正式风险 | 15,128,897.4527 | 17,925,755.5844 | 2,796,858.1317 | 18.4869 | 是 |
| exp002 同确定性 | 15,352,302.2799 | 17,925,755.5844 | 2,573,453.3045 | 16.7627 | 是 |
| exp003 正式 | 14,988,622.6482 | 17,925,755.5844 | 2,937,132.9362 | 19.5957 | 是 |
| exp003 未校准 | 15,258,030.0912 | 17,925,755.5844 | 2,667,725.4932 | 17.4841 | 是 |
| exp004 无季节 | 13,611,587.8329 | 17,925,755.5844 | 4,314,167.7515 | 31.6948 | 是 |
| exp004 历史季节 | 13,626,145.4060 | 17,925,755.5844 | 4,299,610.1784 | 31.5541 | 是 |
| exp004 全年探索 | 13,572,926.4786 | 17,925,755.5844 | — | — | 否 |
| exp001 原登记（不可比） | 16,258,308.2174 | 17,925,755.5844 | — | — | 否 |
| exp006 正式·树DP | 14,066,257.4773 | 17,925,755.5844 | 3,859,498.1072 | 27.4380 | 是 |
| exp006 树DP·种子2026 | 14,055,726.4251 | 17,925,755.5844 | 3,870,029.1593 | 27.5335 | 是 |
| exp006 树DP·种子3407 | 14,075,605.2498 | 17,925,755.5844 | 3,850,150.3346 | 27.3534 | 是 |
| exp006 历史季节·树DP | 14,094,289.9315 | 17,925,755.5844 | 3,831,465.6529 | 27.1845 | 是 |
| exp006 历史季节·2026 | 14,072,183.7562 | 17,925,755.5844 | 3,853,571.8282 | 27.3843 | 是 |
| exp006 历史季节·3407 | 14,086,010.8265 | 17,925,755.5844 | 3,839,744.7579 | 27.2593 | 是 |
| exp006 仅费用 | 14,071,317.0549 | 17,925,755.5844 | 3,854,438.5295 | 27.3922 | 是 |
| exp006 无条件树 | 14,195,217.2998 | 17,925,755.5844 | 3,730,538.2846 | 26.2802 | 是 |
| exp006 较强吞吐惩罚 | 14,079,827.8430 | 17,925,755.5844 | 3,845,927.7414 | 27.3152 | 是 |
| exp006 无终值 | 14,066,257.4773 | 17,925,755.5844 | 3,859,498.1072 | 27.4380 | 是 |
| exp006 旧执行·重新规划 | 13,534,196.7273 | 17,925,755.5844 | 4,391,558.8571 | 32.4479 | 是 |
| exp006 无死区 | 14,065,954.8549 | 17,925,755.5844 | 3,859,800.7295 | 27.4407 | 是 |
| exp006 正式·25kWh | 14,046,310.5439 | 17,925,755.5844 | 3,879,445.0405 | 27.6190 | 是 |
| exp006 历史季节·25kWh | 14,074,930.3326 | 17,925,755.5844 | 3,850,825.2518 | 27.3595 | 是 |
| exp006 固定购电·旧执行 | 13,615,967.6562 | 17,925,755.5844 | 4,309,787.9282 | 31.6525 | 是 |
| exp005 LP·β=0.1 | 14,717,882.1655 | 17,925,755.5844 | — | — | 否 |
| exp005 LP·β=0.01 | 14,740,993.8739 | 17,925,755.5844 | — | — | 否 |
| exp007 轻惩罚·2026 | 17,558,286.5544 | 17,925,755.5844 | 367,469.0300 | 2.0929 | 是 |
| exp007 轻惩罚·3407 | 17,682,555.2532 | 17,925,755.5844 | 243,200.3312 | 1.3754 | 是 |
| exp007 仅费用·42 | 17,725,266.6392 | 17,925,755.5844 | 200,488.9453 | 1.1311 | 是 |
| exp007 仅费用·2026 | 17,627,107.2730 | 17,925,755.5844 | 298,648.3114 | 1.6943 | 是 |
| exp007 仅费用·3407 | 17,537,327.6309 | 17,925,755.5844 | 388,427.9535 | 2.2149 | 是 |

### 实际光伏发电时段预测误差

| 预测来源 | 槽数 | MAE/kW | RMSE/kW | WAPE/% |
|---|---|---|---|---|
| exp001 午夜重评分 | 26,880.0000 | 306.5591 | 455.7890 | 7.2022 |
| exp002 午夜重评分 | 26,880.0000 | 307.1910 | 459.4373 | 7.2171 |
| exp003 正式 | 26,880.0000 | 331.6338 | 501.8496 | 7.7913 |
| exp003 未校准 | 26,880.0000 | 303.6413 | 454.0779 | 7.1337 |
| exp004 无季节 | 26,880.0000 | 379.3326 | 567.1410 | 8.9119 |
| exp004 历史季节 | 26,880.0000 | 377.6413 | 566.5907 | 8.8722 |
| exp006 正式·树DP | 26,880.0000 | 379.3326 | 567.1410 | 8.9119 |
| exp006 历史季节·树DP | 26,880.0000 | 377.6413 | 566.5907 | 8.8722 |
| RL 正式 · 42 | 26,880.0000 | 379.3326 | 567.1410 | 8.9119 |
| RL 纯费用 · 42 | 26,880.0000 | 379.3326 | 567.1410 | 8.9119 |
| exp005 复用无季节预测 | 26,880.0000 | 379.3326 | 567.1410 | 8.9119 |

exp007 六组复用同一 exp004 无季节 seed42 预测，预测误差完全相同，不能声称 RL 提高预测精度。WAPE 原值单位是%，绝对差为百分点；相对变化仍为%。不把不同变量的原始单位强行放在同一坐标。

### 全部阶段时间及计时范围

| 方案 | 阶段 | 累计/s | 组数/日数 | 计时范围 |
|---|---|---|---|---|
| exp001 | 训练 | 5,901.6948 | 165.0000 | 历史训练任务累计；设备/目标数/样本数不同，仅描述 |
| exp001 | 检查点预测 | — | 165.0000 | 历史训练任务累计；设备/目标数/样本数不同，仅描述 |
| exp002 | 训练 | 675.3243 | 33.0000 | 历史训练任务累计；设备/目标数/样本数不同，仅描述 |
| exp002 | 检查点预测 | 3.4018 | 33.0000 | 历史训练任务累计；设备/目标数/样本数不同，仅描述 |
| exp003 | 训练 | 390.4056 | 33.0000 | 历史训练任务累计；设备/目标数/样本数不同，仅描述 |
| exp003 | 检查点预测 | 1.4111 | 33.0000 | 历史训练任务累计；设备/目标数/样本数不同，仅描述 |
| exp004 无季节 | 训练 | 35.3422 | 33.0000 | CPU两线程，33组累计；特征准备另计，不是端到端耗时 |
| exp004 无季节 | 检查点预测 | 0.2235 | 33.0000 | CPU两线程，33组累计；特征准备另计，不是端到端耗时 |
| exp004 历史季节 | 训练 | 34.7130 | 33.0000 | CPU两线程，33组累计；特征准备另计，不是端到端耗时 |
| exp004 历史季节 | 检查点预测 | 0.2259 | 33.0000 | CPU两线程，33组累计；特征准备另计，不是端到端耗时 |
| exp004 全年探索 | 训练 | 36.0251 | 33.0000 | CPU两线程，33组累计；特征准备另计，不是端到端耗时 |
| exp004 全年探索 | 检查点预测 | 0.2230 | 33.0000 | CPU两线程，33组累计；特征准备另计，不是端到端耗时 |
| exp001 同口径重算 | 调度执行 | 19.0684 | 334.0000 | 主种子334日组装求解执行累计，非并行墙钟 |
| exp002 正式风险 | 调度执行 | 907.7402 | 334.0000 | 主种子334日组装求解执行累计，非并行墙钟 |
| exp002 同确定性 | 调度执行 | 17.6989 | 334.0000 | 主种子334日组装求解执行累计，非并行墙钟 |
| exp003 正式 | 调度执行 | 12.3983 | 334.0000 | 主种子334日组装求解执行累计，非并行墙钟 |
| exp003 未校准 | 调度执行 | 12.6689 | 334.0000 | 主种子334日组装求解执行累计，非并行墙钟 |
| exp004 无季节 | 调度执行 | 11.6773 | 334.0000 | 主种子334日组装求解执行累计，非并行墙钟 |
| exp004 历史季节 | 调度执行 | 12.2731 | 334.0000 | 主种子334日组装求解执行累计，非并行墙钟 |
| exp004 全年探索 | 调度执行 | 12.3319 | 334.0000 | 主种子334日组装求解执行累计，非并行墙钟 |
| exp002 | 费用校准 | 197.3803 | — | 历史阶段记录；exp002含四问。缺失不填零 |
| exp003 | 费用校准 | — | — | 历史阶段记录；exp002含四问。缺失不填零 |
| exp004 | 费用校准 | — | 0.0000 | 不适用：未实施费用回选 |
| exp006 正式·树DP | 条件分布准备 | 1.1283 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 正式·树DP | 动态规划 | 7.4953 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 正式·树DP | 实际执行 | 0.0743 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 正式·树DP | 调度执行 | 8.6979 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 正式·树DP | 存储 | 0.1997 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 正式·树DP | 树拟合（含在条件准备） | 0.8012 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 旧执行·重新规划 | 条件分布准备 | 1.3764 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 旧执行·重新规划 | 动态规划 | 9.3332 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 旧执行·重新规划 | 实际执行 | 0.0862 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 旧执行·重新规划 | 调度执行 | 10.7958 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 旧执行·重新规划 | 存储 | 0.2169 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 旧执行·重新规划 | 树拟合（含在条件准备） | 0.9727 | 334.0000 | 单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项 |
| exp006 复用预测 | 训练 | 0.0000 | 0.0000 | 冻结exp004预测，不重复训练；原训练费归exp004 |
| exp006 复用预测 | 检查点预测 | 0.0000 | 0.0000 | 直接读取冻结数组，不重新运行预测网络 |
| RL 正式 · 42 | RL策略训练 | 77.8853 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 正式 · 42 | 历史特征与训练数据准备 | 0.0061 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 正式 · 42 | 检查点存储 | 0.2361 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 正式 · 42 | 本次组墙钟 | 88.3645 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 正式 · 42 | 评价日特征准备 | 1.1273 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 正式 · 42 | 策略推理 | 8.7296 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 正式 · 42 | 逐槽执行 | 0.0826 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 2026 | RL策略训练 | 79.8226 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 2026 | 历史特征与训练数据准备 | 0.0055 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 2026 | 检查点存储 | 0.2356 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 2026 | 本次组墙钟 | 90.4257 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 2026 | 评价日特征准备 | 1.1402 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 2026 | 策略推理 | 8.8716 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 2026 | 逐槽执行 | 0.0828 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 3407 | RL策略训练 | 78.9822 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 3407 | 历史特征与训练数据准备 | 0.0060 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 3407 | 检查点存储 | 0.2338 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 3407 | 本次组墙钟 | 89.4875 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 3407 | 评价日特征准备 | 1.1280 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 3407 | 策略推理 | 8.7889 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 轻惩罚 · 3407 | 逐槽执行 | 0.0822 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 42 | RL策略训练 | 86.4275 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 42 | 历史特征与训练数据准备 | 0.0060 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 42 | 检查点存储 | 0.2551 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 42 | 本次组墙钟 | 98.1310 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 42 | 评价日特征准备 | 1.2670 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 42 | 策略推理 | 9.7842 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 42 | 逐槽执行 | 0.0930 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 2026 | RL策略训练 | 90.3119 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 2026 | 历史特征与训练数据准备 | 0.0065 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 2026 | 检查点存储 | 0.2608 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 2026 | 本次组墙钟 | 102.3961 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 2026 | 评价日特征准备 | 1.3097 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 2026 | 策略推理 | 10.1100 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 2026 | 逐槽执行 | 0.0959 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 3407 | RL策略训练 | 93.3005 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 3407 | 历史特征与训练数据准备 | 0.0064 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 3407 | 检查点存储 | 0.2936 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 3407 | 本次组墙钟 | 106.0030 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 3407 | 评价日特征准备 | 1.3653 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 3407 | 策略推理 | 10.6040 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| RL 纯费用 · 3407 | 逐槽执行 | 0.1009 | — | recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows |
| exp005 LP·β=0.1 | 单组墙钟 | 1,217.5211 | 334.0000 | completed historical arm; solver time included in wall time; different control protocol |
| exp005 LP·β=0.1 | LP求解累计 | 959.6525 | 334.0000 | completed historical arm; solver time included in wall time; different control protocol |
| exp005 LP·β=0.01 | 单组墙钟 | 1,400.2701 | 334.0000 | completed historical arm; solver time included in wall time; different control protocol |
| exp005 LP·β=0.01 | LP求解累计 | 1,131.9447 | 334.0000 | completed historical arm; solver time included in wall time; different control protocol |
| exp007 复用预测 | 新增预测训练 | 0.0000 | 0.0000 | read frozen exp004 tensors; RL policy training recorded separately |

### 历次技术路线如何改变

**exp001**：多网络候选 → 共享四目标早停 → 原协议与同物理重算

**exp002**：固定残差MLP → 风险正式/确定性对照 → 固定午夜购电

**exp003**：问题二独立MLP → 一月费用校准 → 确定性规划+旧因果执行

**exp004**：168小时双分支CNN → 无季节/因果季节 → 沿用exp003规划执行

**exp005**：冻结exp004预测 → 整日场景树两层LP → 硬爬坡与软吞吐罚 → 紧急购电可服务充电

**exp006**：冻结预测 → 条件误差树 → SOC动态规划 → 固定计划+意图裁剪

**exp007**：同预测/条件误差 → PPO有限联合动作 → 固定预算因果训练 → 午夜开环计划+真实执行奖励

![history-costs](figures/history-costs.png)

![history-descriptive-only](figures/history-descriptive-only.png)

![history-errors](figures/history-errors.png)

![stage-timings](figures/stage-timings.png)

## 8. 复现说明

### 核验记录与文件校验值

| 证据文件 | SHA-256 | 状态摘要 |
|---|---|---|
| evidence/independent_qa.json | 067b7fd89afee99880c91d9317804d428a0f7fb1f4f9efc875757d192fe175fc | {"passed": true, "status": "passed", "errors": [], "runs": [{"run_id": "regularized_42", "role": "primary", "rl_seed": 42, "passed": true, "errors": [], "physical_and_billing": {"passed": true, "errors": [], "days": 334, "intervals": 48096, "max_balance_error_kwh": 0.0, "max_soc_error_kwh": 9.094947017729282e-13, "max_cross_day_soc_error_kwh": 0.0, "max_slot_fee_error_yuan": 0.0, "max_daily_cost_error_yuan": 1.4551915228366852e-11, "recomputed_total_cost": 17925755.58441115, "reported_cost_excludes_penalties_and_terminal_value": true, "source_actual_and_tariff_checked": true, "battery_metrics": {"charge_kwh": 3129501.9526110236, "discharge_kwh": 2816762.1744615873, "throughput_kwh": 5946264.127072611, "equivalent_full_cycles": 247.41809443448375, "direction_reversals": 1129, "direction_reversals_including_warmup_boundary": 1129, "simultaneous_slots": 0, "charging_slots": 14760, "discharg |
| evidence/workbook_qa.json | c4ab1ae1b6a3335f97369cfb896065722ab85d7fd813b1aa59329d99a32505df | {"status": "passed", "run_id": "regularized_42", "role": "primary", "physical_and_billing": {"days": 334, "intervals": 48096, "violations": 0, "max_balance_error_kwh": 0.0, "max_state_error_kwh": 9.094947017729282e-13, "minimum_soc_kwh": 1200.0, "maximum_soc_kwh": 10800.0, "max_charge_kwh_per_slot": 833.3333333333334, "max_discharge_kwh_per_slot": 833.3333333333334, "simultaneous_slots": 0, "emergency_charging_slots": 0, "fixed_midnight_purchase_plan": "passed", "projection_check": "passed; vectorized request, balance, SOC and power clipping with 2 kWh deadband", "total_cost_yuan": 17925755.58441115, "planned_cost_yuan": 15770026.245997338, "emergency_cost_yuan": 2155729.3384138113, "charge_kwh": 3129501.9526110236, "discharge_kwh": 2816762.1744615873, "equivalent_full_cycles": 247.41809443448375, "direction_reversals_within_evaluation": 1129, "initial_soc_kwh": 1421.7991105135516, "fina |
| evidence/evidence_manifest.json | e0a29e93ffc58bead55dd029ae9cad3a6624aaa51aba491a7dc6d53d7888cbb9 | {"passed": true, "generator": "reports/exp007_evidence.py", "generator_sha256": "2db046463c7002be670e7cdef47b2d4c2063d786961228a06da9312c0ddb841d", "rows": {"cost_history": 32, "forecast_history": 72, "battery_history": 32, "battery_metrics": 32, "power_variation": 32, "relative_comparison": 732, "timings": 162, "all_runs": 6, "daily": 2004, "historical_daily": 1002, "monthly_cost": 99, "paired_daily": 334, "detail": 5760, "specified": 4, "seed_costs": 6, "seed_summary": 16, "training": 7380, "training_boundaries": 144, "failure_days": 334, "verification": 9, "forecast_monthly": 66, "forecast_lead": 12, "battery_random_days": 5184}, "files": {"all_runs.csv": "adc4296e02b52099f13dc32656f398ea9280fe2033640e013e53e98fb658f77f", "battery/exp004__no_season_full.csv": "0a3234563688b95b46f2fe240497eda1b9f9eb52478656f06d81795bf9aa5525", "battery/exp004__no_season_random_days.csv": "fcfd2f34ee1b9 |
| evidence/exp005_compatibility_audit.json | 7d38a56f5f2f710d26a2b8918d5e8d3c53cd298b5ca542e1e5c1ab3dcca5a36b | {"passed": true, "same_numeric_inputs": true, "same_time_labels": true, "same_forecast_bytes": true, "raw_files": [{"file": "附件1.csv", "current_sha256": "99bae60824c8e80b5b74b180b3f0b76f2649059c437f2a2878d0a7f93e8c598d", "exp005_sha256": "ac5e0ff84e0e339ac6a703d0bb07cec2baf4326fea4025c83f51b0be10e5dabb", "same_bytes": false, "same_table_values_and_time_labels": true, "same_after_crlf_normalization": true}, {"file": "附件2_小区负载.csv", "current_sha256": "a2e0e3e0e35a3551f67a747aeccd308fa745c8a8b6d20bba5d57f05dd9817c97", "exp005_sha256": "4ecde20838ec8c6c1fb890d5cf378fb985d9d2c123fda3bf4458c4d8f6f27b56", "same_bytes": false, "same_table_values_and_time_labels": true, "same_after_crlf_normalization": true}, {"file": "附件2_光伏发电实际功率.csv", "current_sha256": "8889416ff7231b832487a0d1c8629f10cf321a1d7cc5453043b003f864397fce", "exp005_sha256": "7020d6c8f695809b3c8a1730c50294f5c56723652f31e94a43eb132ac |

在独立 RL 工作树根目录重建报告，读取已审核证据，不会启动训练或规划：

```sh
/path/to/python -m reports.exp007_evidence
/path/to/python -m reports.build_report_exp007 --complete --node /path/to/codex/node --data-plugin /path/to/data-analytics
```

独立核验和工作簿导出的完整命令以本包 README、冻结协议和对应导出脚本帮助为准。原始 checkpoint、逐日计划、实际执行、训练诊断及签名在 `data/results/exp007/` 中。重新训练使用新的隔离副本，不覆盖本次原始计时。

报告正文、图表、离线HTML和工作簿从同源冻结结果产生；显示时四舍五入，CSV和数组保留核算精度。HTML内嵌全部交互数据；工作簿和证据下载依赖同目录附件，请保留整个报告包。

完整六组正式任务墙钟为 **574.90 秒**；主组墙钟 **88.36 秒**，其中策略训练 **77.89 秒**。执行源码提交为 `d5ed43b93f27db4fe30d49505a817d13a096f843`，独立核验已逐文件对照冻结Git源码。

复现入口为：

```sh
/path/to/python -m experiments.problem2.rl_planning.run
/path/to/python -m experiments.problem2.rl_planning.verify
/path/to/python -m experiments.problem2.rl_planning.export --node /path/to/codex/node
```

执行前阅读各入口帮助及本包README；完整重训在新隔离副本进行，已有结果保持原始时间记录。独立核验覆盖6组、每组334日×144槽、固定训练预算、历史截止、checkpoint、锁定计划、真实物理执行与费用；`passed=True`，错误条数 `0`。

正式构建前已核对远程 main 为 `bca8dbe3004b0a87cd7be71a06891a4af87a9eb8`，其 `reports/` 与本地模板提交 `7ecd9b988665ad628dbf048ed295457166170ffc` 无差异。报告保留八节结构、新增随机日电池原始曲线、完整334日原始曲线和波动诊断；模板逐文件SHA见[报告契约](report_contract.md)。

代码构建与实际浏览器视觉验收分开记录。独立物理/费用/因果及工作簿核验以同包证据为准；浏览器截图、桌面/窄屏布局与代表性交互检查由最终验收记录另行确认，不以构建成功代替。
