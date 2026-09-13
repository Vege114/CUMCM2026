# exp008 最终成果

[离线报告](report.html) · [Markdown正文](report.md) · [指定日期完整表](specified_dates.md) · [正式记录](record.json)

本版从 exp006 分支出发，完成第一至第四问的模型与规划迭代。用户接受当前已验证结果后停止优化并定稿。第二问334日实际费用为13,201,981.95元，比exp006下降6.1443%；非空方向反转从2729降至2533。原8%目标没有达到，接受记录没有改变该事实。部分行为指标增加，详见正文。

第一问、第二问、第三问、第四问两个方案均保留独立冻结来源。第三、4-3问采用单HGB与官方PV更新，4-2采用单HGB和联动价格，不能宣称四问全部共享第二问最终融合预测器。

五份按题目附件格式填写的成果：

- [result1.xlsx](result1.xlsx)
- [result2.xlsx](result2.xlsx)
- [result3.xlsx](result3.xlsx)
- [result4-2.xlsx](result4-2.xlsx)
- [result4-3.xlsx](result4-3.xlsx)

核验入口：[最终选择](../../../experiments/exp008/final_selection.json)、[最终数据载荷](final_payload.json)、[物理与计费独立审计](evidence/final_selection_audit.json)、[电池原始轨迹](evidence/battery/README.md)、[预测证据](evidence/forecast/README.md)、[工作簿回读和页面检查](evidence/workbooks/README.md)。图表同时提供PNG和SVG；年度电池图保留全部48096个原始时点。

报告使用main提交7ecd9b988665ad628dbf048ed295457166170ffc中的八节模板。只查看正文、离线网页、图和Excel不需要训练环境。重建报告使用冻结数组，不重训练、不重新优化；完整命令和依赖在正文第8节及[evidence/reproduction.json](evidence/reproduction.json)。未选中间候选和重复月度权重不是本版正式结果。
