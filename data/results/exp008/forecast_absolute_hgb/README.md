# 唯一直接绝对值 HGB 候选与固定 LP 桥接

预先保存的协议为 `protocol.json`。两个独立 HGB 分别预测绝对负载/PV，全年输入语义一致；排除旧 HGB 的 base_load/base_pv/base_net/base_is_cnn 四列，保留 31 列历史实际与日历特征。没有 CNN 或周期预测输入、附件 3、未来价格或全年季节形状。输出非负，PV 使用原有过去 28 日日照并集扩展 20 分钟的因果遮罩。

固定 max_leaf_nodes=15、max_iter=100、minleaf=50、l2=2、learning_rate=0.08、seed=42、90 日年龄权重。每月过去数据重新训练，最后 7 个完整日作为显式验证集；固定 patience=10 早停，未按正式月分数选迭代或参数。22 份完整 joblib 模型、逐月训练/验证边界、334 日预测与 hash 均保留。与旧 net-HGB 相比同时改变标签、去掉参考特征、采用双模型及验证早停，不能称单因素试验。

固定主比较是 HGB → 同 Ridge28 → 同非递归 0.5 memory，与 Joint CNN → 同 Ridge28 → 同 0.5 memory 比较。原始和仅 Ridge28 两层也完整保存，没有按年度分数选择后处理。

| 指标 | Joint + Ridge28 + Memory | HGB + Ridge28 + Memory |
|---|---:|---:|
| 净负载 RMSE / kW | 375.585 | 362.899 |
| 高价时段净 RMSE / kW | 358.230 | 348.184 |
| 价格加权净 RMSE / kW | 377.212 | 364.951 |
| 日净能量 RMSE / kWh | 3723.897 | 3610.633 |
| 日内累计净误差 RMSE / kWh | 2351.532 | 2285.937 |

三类五项预设门槛全部改善，`dispatch_gate.json` 为 true，才执行同 tree28/q0.8/state_buffer500 全 334 日 LP 桥接。新的完整账单为 **13,173,407.796092 元**，计划费 12,593,868.384542 元、紧急费 579,539.411549 元，比原 Memory 桥接 **再省 81,341.564431 元**。其中日前少 60,483.638334 元，紧急少 20,857.926097 元。相对 exp006 降费 6.34746%，仍未达到 8%。

该桥接使用原本的频繁 greedy/state buffer 执行，只用于判断预测价值：非空换向 6983、active 40268、吞吐 12105874.684348 kWh、同充放 0；它不是低换向最终策略。物理/费用独立核验通过，起始 SOC 1421.7991105135516 kWh，末 SOC 1570.5801961587242 kWh。

原始 HGB 的净 RMSE 433.740，略差于原始 Joint 的 431.122，日净能量误差也更大。改善发生在事先固定的校准与记忆完整管线，不能宣传为未经校准的 HGB 全面优于网络。所有数字来自已反复查看的 2025 开发年，不是独立测试集。

验证包含 22 模型保存重载一致、7 个发行点的未来实际/未来预测污染、2 月与 9 月在月初后数据被污染条件下完整重训的输出一致、训练/验证/正式月边界、同一既有 Memory 对照数值一致，以及完整 LP 的原始附件核费/物理。见 `causality_verification.json`、`independent_verification.json`、`lp_bridge/.../independent_verification.json`。

后续可用 `experiments.exp008.forecast_absolute_hgb.AbsoluteHGBStore` 读取签名主档案；`experiments.exp008.absolute_hgb_forecast_adapter.AbsoluteHGBForecasts` 或 `audited_forecasts()` 提供 issue-time/历史误差与可审计 override 接口。未选择最终模型，未生成最终报告。
