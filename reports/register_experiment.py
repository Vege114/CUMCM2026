"""Register a measured experiment once and refresh comparisons with all prior runs."""

import argparse
import csv
import json
import re
from pathlib import Path

from history import comparison_rows, read_registry


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', required=True, type=Path)
    args = p.parse_args()
    current = json.loads(args.record.read_text())
    root = Path(__file__).resolve().parent
    schema = json.loads((root/'templates/experiment.schema.json').read_text())
    for key in schema['required']:
        if key not in current or current[key] in (None, {}, [], ''):
            p.error(f'必须填写真实实验记录字段：{key}')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', current['experiment_id']):
        p.error('实验编号格式错误。')
    if not re.fullmatch(r'[0-9a-f]{40}', current['code_commit']):
        p.error('code_commit 必须是实验代码的完整 Git 提交。')
    for key in schema['properties']['protocol']['required']:
        if key not in current['protocol']:
            p.error(f'缺少评价协议字段：{key}')
    encoded = json.dumps(current,ensure_ascii=False,indent=2,allow_nan=False)
    registry = root/'registry'; registry.mkdir(exist_ok=True)
    destination = registry/f"{current['experiment_id']}.json"
    if destination.exists():
        p.error('实验编号已经登记。旧记录不可覆盖，请使用新的实验编号。')
    prior = read_registry(registry)
    relative = comparison_rows(prior,current)
    report_path=root/current['artifacts']['report']
    if not report_path.is_file():p.error('必须先创建正文文件，才能登记实验。')
    history_text=(f'本次与此前 {len(prior)} 次实验比较。' if prior else
                  '首次正式实验，当前没有历史训练成绩。历史同期预测属于本轮对照。')
    history_text+='\n\n| 实验 | 问题 | 总费用（元） | 技术路径 |\n|---|---|---:|---|\n'
    for record in prior+[current]:
        route=' → '.join(record['technical_path']).replace('|','\\|')
        for metric in record['metrics']:
            history_text+=f"| {record['experiment_id']} | {metric['scenario']} | {metric['total_cost']:,.4f} | {route} |\n"
    history_text+='\n[全部预测、调度指标及相对变化](relative_comparison.csv)。不同评价口径的记录不计算直接排名，差异见该文件。'
    body=report_path.read_text()
    if '{{history_comparison}}' in body:
        body=body.replace('{{history_comparison}}',history_text)
    else:
        body+='\n\n### 自动生成的全部历史比较\n\n'+history_text
    report_path.write_text(body)
    destination.write_text(encoded)
    columns = ['previous_experiment','current_experiment','task','route','metric','previous',
               'current','relative_change_pct','comparison','previous_technical_path','current_technical_path']
    for path in (root/'relative_comparison.csv',args.record.parent/'relative_comparison.csv'):
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(relative)
    index = '# 实验报告索引\n\n'+'\n'.join(
        f"- [{r['experiment_id']}：{r['title']}]({r['artifacts']['report']})" for r in prior+[current])
    index += '\n\n[本次与此前全部实验的逐项相对变化](relative_comparison.csv)\n\n负值表示该指标数值下降；期末储电量和耗时应分别解释，不能一概视为收益。\n'
    (root/'latest.md').write_text(index)
    print(f"已登记 {current['experiment_id']}，与 {len(prior)} 次历史实验进行逐项比较。")


if __name__ == '__main__': main()
