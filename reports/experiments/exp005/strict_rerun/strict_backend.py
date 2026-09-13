"""Add exact node exclusivity at the solver boundary; original files stay intact."""

import warnings
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, hstack, vstack

from experiments.problem2.stochastic_lp import model


class StrictBackend:
    """Same two original objectives, plus c <= M*z, d <= M*(1-z)."""

    def __init__(self):
        self.logs = []
        self.last_problem = None
        self.last_result = None

    def linprog(self, objective, *, A_ub, b_ub, A_eq, b_eq, bounds, method, options):
        nodes = len(b_eq) // 2
        old_size = len(objective)
        horizon = old_size - 6 * nodes
        charge = horizon + np.arange(nodes)
        discharge = horizon + nodes + np.arange(nodes)
        switch = old_size + np.arange(nodes)
        size = old_size + nodes
        upper = np.asarray(bounds)[charge, 1]
        assert horizon > 0 and np.isfinite(upper).all()
        assert np.array_equal(upper, np.asarray(bounds)[discharge, 1])
        exclusive = coo_matrix((np.r_[np.ones(nodes), -upper, np.ones(nodes), upper],
                                (np.r_[np.arange(nodes), np.arange(nodes), nodes + np.arange(nodes), nodes + np.arange(nodes)],
                                 np.r_[charge, switch, discharge, switch])), shape=(2 * nodes, size)).tocsr()
        zero_eq = coo_matrix((A_eq.shape[0], nodes))
        zero_ub = coo_matrix((A_ub.shape[0], nodes))
        matrix = vstack([hstack([A_eq, zero_eq]), hstack([A_ub, zero_ub]), exclusive]).tocsc()
        lower = np.r_[b_eq, np.full(len(b_ub) + 2 * nodes, -np.inf)]
        rhs = np.r_[b_eq, b_ub, np.zeros(nodes), upper]
        full_bounds = np.vstack([bounds, np.tile([0., 1.], (nodes, 1))])
        integrality = np.r_[np.zeros(old_size, dtype=int), np.ones(nodes, dtype=int)]
        cost = np.r_[objective, np.zeros(nodes)]
        self.last_problem = {'objective': cost, 'matrix': matrix, 'lower': lower, 'upper': rhs,
                             'bounds': full_bounds, 'integrality': integrality}
        began = perf_counter()
        # An invalid numerical candidate is discarded and the SAME stage is
        # resolved at tighter feasibility, within its original 60-second cap.
        # There is no extra objective, physical projection, or relaxed check.
        attempts = []
        for tolerance in [1e-6, 1e-10]:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Unrecognized options detected:.*', category=RuntimeWarning)
                result = milp(cost, integrality=integrality, bounds=Bounds(*full_bounds.T),
                              constraints=LinearConstraint(matrix, lower, rhs), options={
                                  'time_limit': max(.001, options['time_limit'] - (perf_counter() - began)),
                                  'mip_rel_gap': 0.0, 'mip_feasibility_tolerance': tolerance})
            residual = None
            if result.success:
                lhs = matrix @ result.x
                residual = max(float(np.maximum(lower - lhs, 0).max()),
                               float(np.maximum(lhs - rhs, 0).max()),
                               float(np.maximum(full_bounds[:, 0] - result.x, 0).max()),
                               float(np.maximum(result.x - full_bounds[:, 1], 0).max()),
                               float(np.minimum(result.x[charge], result.x[discharge]).max()),
                               float(np.abs(result.x[switch] - np.rint(result.x[switch])).max()))
            attempts.append({'status': int(result.status), 'mip_feasibility_tolerance': tolerance,
                             'max_physical_or_integrality_residual': residual})
            if not result.success or residual <= 1e-6:
                break
        log = {'status': int(result.status), 'message': result.message,
               'seconds': perf_counter() - began, 'variables': size, 'integer_variables': nodes,
               'numerical_attempts': attempts, 'numerical_retry_count': len(attempts) - 1,
               'mip_gap': None if result.get('mip_gap') is None else float(result.mip_gap),
               'mip_dual_bound': None if result.get('mip_dual_bound') is None else float(result.mip_dual_bound),
               'mip_node_count': None if result.get('mip_node_count') is None else int(result.mip_node_count)}
        if result.x is not None:
            log['max_binary_residual'] = float(np.max(np.abs(result.x[switch] - np.rint(result.x[switch]))))
            log['max_exclusivity_residual'] = float(np.maximum(exclusive @ result.x - np.r_[np.zeros(nodes), upper], 0).max())
            log['max_overlap_kwh'] = float(np.minimum(result.x[charge], result.x[discharge]).max())
            self.last_result = result
            if result.success and (log['max_binary_residual'] > 1e-6
                                   or log['max_exclusivity_residual'] > 1e-6
                                   or log['max_overlap_kwh'] > 1e-6):
                self.logs.append(log)
                raise model.SolveError(f'Strict mutual exclusion failed numerical audit: {log}')
            # Do not round or clean the physical solution returned to original code.
            result.x = result.x[:old_size]
        self.logs.append(log)
        self.last_result = result
        return result

    def solve_tree(self, *args, **kwargs):
        self.logs = []
        with patch.object(model, 'linprog', self.linprog):
            result = model.solve_tree(*args, **kwargs)
        for original, strict in zip(result['metadata']['stages'], self.logs, strict=True):
            original.update(strict)
        result['metadata'].update(model_class='MILP', strict_mutual_exclusion=True,
                                  exclusivity='c <= M*z; d <= M*(1-z); z binary', mip_rel_gap_requested=0.0,
                                  physical_acceptance_tolerance=1e-6,
                                  integer_tolerance_retry=[1e-6, 1e-10])
        return result
