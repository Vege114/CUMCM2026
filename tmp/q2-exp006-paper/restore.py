from pathlib import Path
import re
root=Path(__file__).resolve().parents[2]
work=root/'tmp/q2-exp006-paper'
# Restore published four-decimal values from retained rendered pages 23--26.
days=[
('3月20日','20250320',[8.4117,679.3372,0,462.4444,387.6183,27.4365],[70346.8576,44611.7234,1716.3069,1548.9025],[(7897.4715,0),(2371.7082,5828.0182),(5533.9859,2670.2740),(3488.0281,246.2828),(0,6026.3271),(632.4555,3319.2446)]),
('6月21日','20250621',[0,27.7165,0,235.8488,31.5057,6.3675],[38948.4076,23451.0853,3021.5486,2896.9393],[(0,0),(1001.3879,1748.7532),(5533.9859,0),(3606.8930,0),(0,5144.3424),(685.1602,2969.8034)]),
('9月23日','20250923',[0,424.6160,0,593.6816,800.2914,9.7287],[68715.0583,44341.4090,1792.1947,1693.4148],[(7908.8883,0),(1633.8435,6405.9719),(4825.9760,1792.3856),(4575.4044,327.0221),(0,5531.0383),(685.1602,3704.6345)]),
('12月21日','20251221',[3.4015,916.0779,0,823.6368,523.3691,22.2234],[96470.3643,62775.1315,2083.6162,1644.8034],[(7601.5442,0),(3336.0342,2381.1137),(5176.8954,7179.0131),(6400.2571,2434.0434),(0,5964.7846),(632.4555,3289.8075)])]
prefix='\\begin{table}[H]\n\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.13}\n'
end='\\bottomrule\\end{tabularx}\n\\end{table}\n'
def row(c): return ' & '.join(c)+r'\\'+'\n'
def num(x):return f'{x:.4f}'
tabs=[]
for i,(date,stamp,p,tot,b) in enumerate(days):
    s=('\\clearpage\n' if i==2 else '')+prefix+rf'\caption{{2025年{date}指定时段购电量及全天费用}}\label{{tab:q2-purchase-{stamp}}}'+'\n'+r'\begin{tabularx}{\textwidth}{CrCrCr}'+'\n'+r'\toprule'+'\n'+row(['时间段','购电量']*3)+r'\midrule'+'\n'
    for indices in ([0,1,2],[3,4,5]):
        vals=[]
        for k in indices: vals.extend([f'{10+2*k}:00--{10+2*k}:10',num(p[k])])
        s+=row(vals)
    s+=r'\midrule'+'\n'+row([r'\multicolumn{2}{c}{全天计划购电量}',num(tot[0]),r'\multicolumn{2}{c}{全天购电费}',num(tot[1])])+end
    s+=prefix+rf'\caption{{2025年{date}储能设备实际充放电量与储电量}}\label{{tab:q2-battery-{stamp}}}'+'\n'+r'\begin{tabularx}{\textwidth}{CrrCrr}'+'\n'+r'\toprule'+'\n'+row(['时间段','充电量','放电量']*2)+r'\midrule'+'\n'
    for indices in ([0,1],[2,3],[4,5]):
        vals=[]
        for k in indices:vals.extend([f'{4*k}:00--{4*k+4}:00',num(b[k][0]),num(b[k][1])])
        s+=row(vals)
    s+=r'\midrule'+'\n'+row([r'\multicolumn{2}{c}{0:00储电量}',num(tot[2]),r'\multicolumn{2}{c}{24:00储电量}',num(tot[3])])+end
    tabs.append(s)
cols=[
 [('01:30--01:40',2.3505),('03:10--03:30',25.5499),('03:50--04:00',.2329),('04:50--05:00',3.9933),('15:20--16:20',374.1134),('16:30--17:00',36.3404),('17:20--17:30',.2045),('18:40--19:00',27.3934),('20:00--20:20',27.1328),('21:10--21:40',15.8113),('23:30--23:50',3.6149)],
 [('02:50--03:00',8.8969),('04:00--04:10',8.7379),('05:00--05:10',.4800),('05:40--05:50',1.5465),('06:10--06:20',9.3005)],
 [('02:10--02:20',5.2487),('05:00--05:10',9.4808),('09:30--09:50',48.0591),('10:00--10:10',3.3890),('14:20--14:30',7.5128),('14:40--15:00',114.1363),('18:10--18:20',11.2574),('20:00--20:20',38.1112),('20:50--21:00',37.9974),('21:20--21:30',16.4217),('22:20--22:30',5.8906)],
 [('01:30--01:40',5.8174),('03:00--03:10',8.7709),('03:30--03:50',62.2101),('07:50--08:00',18.8978),('10:30--10:40',47.4418),('16:50--17:00',12.6858),('17:20--17:30',12.0573),('18:30--18:50',11.7931),('19:50--20:00',15.9321),('23:30--23:40',1.0964)]
]
em=prefix+r'\caption{四个指定日期的全部紧急购电结果（kWh）}\label{tab:q2-emergency-all}'+'\n'+r'\begin{tabularx}{\textwidth}{CrCrCrCr}'+'\n'+r'\toprule'+'\n'+row([r'\multicolumn{2}{c}{'+s+'}' for s in ['2025.03.20','2025.06.21','2025.09.23','2025.12.21']])+r'\midrule'+'\n'+row(['时间段','购电量']*4)
for i in range(11):
    v=[]
    for col in cols:v.extend([col[i][0],num(col[i][1])] if i<len(col) else ['--','--'])
    em+=row(v)
em+=r'\midrule'+'\n'+row(['合计','516.7373','合计','28.9619','合计','297.5050','合计','196.7028'])+end
annual=prefix+r'\caption{评价期真实费用与电池运行指标对照}\label{tab:q2-annual}'+'\n'+r'\begin{tabularx}{\textwidth}{Xrrr}'+'\n'+r'\toprule'+'\n'+row(['指标','同预测基础调度','正式主组','旧执行重规划对照'])+r'\midrule'+'\n'
for vals in [('计划购电/万kWh','2099.7148','2170.2244','2149.8222'),('总费用/万元','1361.1588','1406.6257','1353.4197'),('紧急购电/kWh','160912.1654','197559.7826','74976.2978'),('非空方向反转/次','7715','2729','6755'),('电芯等效满循环','499.1216','477.7055','489.3580'),('功率总变差/万kW','3343.0433','3045.1039','3266.6146')]:annual+=row(vals)
annual+=end
m=(work/'methods.tex').read_text(encoding='utf-8')
for title in ['指定日期购电及储能结果','紧急购电与评价期绩效']:m=m.replace('\\subsection{'+title+'}','\\clearpage\n\\subsection{'+title+'}')
m=m.replace('% Q2_SPECIFIED_TABLES','\n'.join(tabs)).replace('% Q2_EMERGENCY_TABLE',em).replace('% Q2_ANNUAL_TABLE',annual)
tex=root/'Xelatex/数模通用模板.tex'
old=tex.read_text(encoding='utf-8')
a=old.index(r'\section{问题二的模型的建立和求解}')
b=old.index(r'\section{问题三的模型的建立和求解}',a)
new=old[:a]+m+'\n'+old[b:]
assert new[:a]==old[:a] and new[new.index(r'\section{问题三的模型的建立和求解}'):]==old[b:]
tex.write_text(new,encoding='utf-8',newline='\r\n')
(work/'restored-full-section.tex').write_text(m,encoding='utf-8')
(work/'recovery-note.md').write_text('本次从保留的 methods.tex 和 page-23 至 page-26 渲染页恢复正文及四位小数表格。原 exp006 目录当前缺失；未重新读取工作簿，未重新运行实验。历史 verification.json 为上轮核验记录。\n',encoding='utf-8')
print('Restored',len(m),'characters;',m.count(r'\begin{table}'),'tables. Other sections unchanged.')
