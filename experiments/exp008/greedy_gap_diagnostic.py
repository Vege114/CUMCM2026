"""Read-only comparison of archived scenario recourse and causal greedy replay.

These are optimization-surrogate diagnostics, not eligible evaluation bills.
No future execution path is used to construct a purchase or controller.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from experiments.exp008.closed_loop import objective


def diagnose(directory):
    directory = Path(directory)
    rows = []
    with np.load(directory/'dispatch.npz') as dispatch:
        for i, day in enumerate(dispatch['days']):
            with np.load(directory/f'planning_day{day}.npz') as pack:
                price = pack['price']
                selected = pack['all_net_paths'][pack['selected_scenario_indices']]
                soc = float(pack['initial_soc'])
                terminal = 0. if day == 364 else price.min()/np.sqrt(.9)
                q, mask = pack['purchase'], pack['allowed_charge']
                recourse = float(q @ price + np.mean((5*price*pack['scenario_emergency']
                    + .002*(pack['scenario_charge']+pack['scenario_discharge'])).sum(axis=1))
                    - terminal*np.mean(pack['scenario_states'][:,-1]-1200.))
                selected_greedy = objective(q, selected, price, soc, charge_mask=mask,
                    throughput=.002, variation=0., terminal=terminal)[0]
                raw_all = objective(q, pack['all_net_paths'], price, soc, charge_mask=mask,
                    throughput=.002, variation=0., terminal=.45 if day!=364 else 0.)[0]
                refined_all = objective(dispatch['original'][i], pack['all_net_paths'], price, soc,
                    charge_mask=mask, throughput=.002, variation=0., terminal=.45 if day!=364 else 0.)[0]
                rows.append(dict(day=int(day), scenario_recourse_objective=recourse,
                    selected_greedy_objective=selected_greedy,
                    recourse_optimism=selected_greedy-recourse,
                    initial_all_history_objective=raw_all,
                    refined_all_history_objective=refined_all,
                    historical_refinement_improvement=raw_all-refined_all))
    frame = pd.DataFrame(rows)
    out = Path('data/results/exp008/greedy_gap_diagnostic')/directory.name
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'daily.csv', index=False)
    result = {'role': 'historical-scenario surrogate diagnostic, not actual evaluation',
              'input_directory': str(directory.resolve()), 'days': len(frame),
              'sum': frame.drop(columns='day').sum().to_dict(),
              'largest_optimism_days': frame.nlargest(5,'recourse_optimism').to_dict('records')}
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    diagnose('data/results/exp008/mode_planning_physical/joint_ridge28_refined_s3_30days')
