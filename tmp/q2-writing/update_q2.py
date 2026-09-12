from pathlib import Path
import json

root = Path(__file__).resolve().parents[2]
tex = root / 'Xelatex/数模通用模板.tex'
before = tex.read_bytes()
text = before.decode('utf-8')
newline = '\r\n' if '\r\n' in text else '\n'
start = text.index(r'\section{问题二的模型建立与求解}')
end = text.index(r'\section{问题三的模型的建立和求解}', start)
section = (root / 'tmp/q2-writing/q2-section.tex').read_text(encoding='utf-8')
section = section.replace('\r\n', '\n').rstrip() + '\n\n'
section = section.replace('\n', newline)
updated = text[:start] + section + text[end:]
# Synchronize only the earlier paragraph explicitly devoted to problem two.
a = updated.index(r'\subsection*{问题2分析}')
b = updated.index(r'\subsection*{问题3分析}', a)
analysis = r'''\subsection*{问题2分析}
日前购电必须在实际供需揭晓前确定，而计划剩余与供电不足的费用不对称。先以历史同期曲线和卷积残差网络预测负载、光伏，再由条件误差决策树构造逐槽净需求支持，采用离散储能动态规划权衡计划购电、缺口风险及操作强度。实际运行时锁定购电量，允许电池按当槽余缺改变充放电方向，并连续传递真实储电量。最后分别核算购电费用、吞吐、反转和功率变化，通过同预测输入下的规划及执行对照评价模型。

'''.replace('\n', newline)
updated = updated[:a] + analysis + updated[b:]
tex.write_bytes(updated.encode('utf-8'))

def outside_q2(s):
    a = s.index(r'\subsection*{问题2分析}')
    b = s.index(r'\subsection*{问题3分析}', a)
    s = s[:a] + '<Q2_ANALYSIS>' + s[b:]
    a = s.index(r'\section{问题二的模型建立与求解}')
    b = s.index(r'\section{问题三的模型的建立和求解}', a)
    return s[:a] + '<Q2_SECTION>' + s[b:]

assert outside_q2(text) == outside_q2(updated), 'Unrelated content changed'
receipt = {'modified_only_q2_body_and_q2_analysis': True,
           'before_bytes': len(before), 'after_bytes': len(updated.encode('utf-8'))}
(root / 'tmp/q2-writing/edit-scope.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt))
