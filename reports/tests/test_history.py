import copy
import unittest

from reports.history import comparison_rows, differences


class HistoryTests(unittest.TestCase):
    def record(self, name, cost):
        return {'experiment_id': name, 'protocol': {'version': 'v1', 'period': ['a','b'],
                'variants': [name]}, 'data_hashes': {'data':'same'},
                'technical_path': [name], 'metrics': [{'scenario': '2', 'total_cost': cost}]}

    def test_compares_all_prior_experiments_and_allows_new_models(self):
        prior = [self.record('one', 100.), self.record('two', 200.)]
        before = copy.deepcopy(prior)
        current = self.record('three', 90.)
        rows = comparison_rows(prior, current)
        self.assertEqual([r['relative_change_pct'] for r in rows], [-10., -55.])
        self.assertEqual(prior, before)
        self.assertFalse(differences(prior[0], current))

    def test_different_billing_is_not_ranked(self):
        a, b = self.record('one', 100.), self.record('two', 90.)
        b['protocol']['billing'] = 'different'
        rows = comparison_rows([a], b)
        self.assertEqual(rows[0]['comparison'], 'billing')
        self.assertIsNone(rows[0]['relative_change_pct'])

    def test_primary_forecast_role_matches_across_architectures(self):
        a,b=self.record('old',100.),self.record('new',90.)
        a['forecast_metrics']=[{'variant':'old_network','role':'primary','target':'load','population':'all','mae':10.}]
        b['forecast_metrics']=[{'variant':'new_network','role':'primary','target':'load','population':'all','mae':8.}]
        b['protocol']['efficiency']={'charge':.95}
        rows=comparison_rows([a],b)
        self.assertIsNone(rows[0]['relative_change_pct'])
        self.assertAlmostEqual(rows[1]['relative_change_pct'],-20.)

    def test_changed_pv_integration_blocks_forecast_ranking(self):
        a,b=self.record('old',100.),self.record('new',90.)
        for record in (a,b):record['forecast_metrics']=[{'variant':'primary','target':'pv_corrected','population':'all','mae':10.}]
        b['protocol']['pv_interpolation']='integrated'
        rows=comparison_rows([a],b)
        self.assertIsNone(rows[-1]['relative_change_pct'])
        self.assertIn('pv_interpolation',rows[-1]['comparison'])

    def test_legacy_primary_selection_keeps_question_identity(self):
        a, b = self.record('old', 100.), self.record('new', 90.)
        a['forecast_metrics'] = [
            {'variant': 'selected_2', 'scenario': '2', 'target': 'load', 'population': 'all', 'mae': 10.},
            {'variant': 'selected_3', 'scenario': '3', 'target': 'load', 'population': 'all', 'mae': 20.}]
        b['forecast_metrics'] = [
            {'variant': 'new_mlp', 'role': 'primary', 'scenario': '2', 'target': 'load', 'population': 'all', 'mae': 8.},
            {'variant': 'new_mlp', 'role': 'primary', 'scenario': '3', 'target': 'load', 'population': 'all', 'mae': 8.}]
        before = copy.deepcopy(a)
        rows = comparison_rows([a], b)
        self.assertEqual([r['relative_change_pct'] for r in rows[1:]], [-20., -60.])
        self.assertEqual(a, before)


if __name__ == '__main__': unittest.main()
