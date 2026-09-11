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


if __name__ == '__main__': unittest.main()
