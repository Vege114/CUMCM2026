## 6. 结果及失败案例

![费用组成对照（万元）](figures/q2-cost-components.png)

[论文SVG](figures/q2-cost-components.svg)

![逐月购电费用（万元）](figures/q2-monthly-cost.png)

[论文SVG](figures/q2-monthly-cost.svg)

![逐月凌晨预测误差](figures/q2-monthly-error.png)

[论文SVG](figures/q2-monthly-error.svg)

![指定日购电与净需求](figures/q2-specified-days.png)

[论文SVG](figures/q2-specified-days.svg)

正式策略最贵日为2025-07-03，费用154,222.41元。以下供需、购电与SOC取自该日真实回放，作为事后失败诊断；该日不参与重新选参。费用尾部CVaR90按全年最贵10%概率质量计算，包括边界日的部分权重。

![最贵日2025-07-03：净需求与购电](figures/q2-worst-day.png)

[论文SVG](figures/q2-worst-day.svg)
