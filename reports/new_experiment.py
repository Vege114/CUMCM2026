"""Create an unregistered report draft without inventing measurements."""

import argparse
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', required=True)
    p.add_argument('--title', required=True)
    args = p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.id):
        p.error('实验编号只能包含英文字母、数字、下划线和连字符。')
    root = Path(__file__).resolve().parent
    destination = root / 'experiments' / args.id
    destination.mkdir(parents=True, exist_ok=False)
    template = (root / 'templates/report.md').read_text()
    (destination / 'report.md').write_text(template.replace('{{experiment_id}}', args.id).replace('{{title}}', args.title))
    draft = {'experiment_id':args.id,'title':args.title,'protocol':{},'code_commit':'',
             'data_hashes':{},'environment':{},'seeds':[],'models':[],
             'model_configuration':{},'metric_definitions':{},'metrics':[],
             'forecast_metrics':[],'technical_path':[],
             'artifacts':{'report':f'experiments/{args.id}/report.md',
                          'html':f'experiments/{args.id}/report.html'}}
    (destination/'record.draft.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2))
    print(destination)


if __name__ == '__main__': main()
