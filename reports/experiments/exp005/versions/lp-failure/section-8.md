## 8. 复现说明

运行 `.venv/bin/python -m unittest discover -s tests -p test_q2_stochastic_lp.py -v` 检查11项模型测试。运行 `.venv/bin/python -m experiments.problem2.stochastic_lp.run --days 2 --out .work/exp005-reproduce` 可复现首日停止，输出目录必须不存在。

运行 `.venv/bin/python reports/build_report_exp005.py` 重建证据、静态图和正文；构建器在完全相同输入和原模型下复算失败窗口，保存两层数组并核对原结果，没有改变目标或约束。网页使用与既有报告相同的Data共享运行时，数据完整内嵌到离线HTML。

证据包包括失败记录、固定协议、午夜144段计划、停止前81段实际轨迹、13:30失败窗口的两层完整数组、CSV明细、独立审计、测试日志和源文件哈希。失败报告整理完成不等于购电策略通过认证。
