# 本轮独立审稿结果

基线：88d6b160。W1、源码审查、远程两图数据核验、最终PDF视觉和扫描均PASS，详见gate_summary.json。最终原生PDF为133页、正文27页、附录29页起；SHA256为45c5be4400860e5c163de5588fa45c24c2f1c0573f5c39b2ed7e72deefea83b3。

源复验：`.venv/bin/python reports/experiments/exp009/frontmatter_layout_audit/verify_source.py`。

图件与PDF复验：用安装pdfplumber、pypdf的Python分别执行同目录verify_remote_visuals.py、verify_pdf.py。PDF脚本默认为Xelatex/数模通用模板.pdf；须先从最新源编译，图件缺失时明确中止，防止误审旧PDF。编译与规范检查由主任务负责。

文件保护、公式/表格/引用、两图逐行数值映射、实际页码和人工视觉记录分见相应JSON与visual_review.md。五份新远程check PDF只取消跟踪、保留本地并加入ignore，记录见build_artifact_verification.json。审稿者未修改被审正文或模型，未执行优化或预测训练。

基线记录内98da页数仅为早期参考，不支持对最新远程版本计算精确净省页。旧单幅敏感性图及旧PDF检查不再纳入当前验收。
