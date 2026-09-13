"""Same strict model and physical checks, with a user-proposed MIP gap."""

from unittest.mock import patch

import strict_backend
from scipy.optimize import milp as scipy_milp


class GapBackend(strict_backend.StrictBackend):
    def __init__(self, relative_gap):
        super().__init__()
        if relative_gap not in (0., .0005, .001):
            raise ValueError('Benchmark only the exact, 0.05%, and 0.1% settings')
        self.relative_gap = relative_gap

    def solve_tree(self, *args, **kwargs):
        def with_gap(*solver_args, **solver_kwargs):
            options = dict(solver_kwargs['options'])
            options['mip_rel_gap'] = self.relative_gap
            solver_kwargs['options'] = options
            return scipy_milp(*solver_args, **solver_kwargs)

        with patch.object(strict_backend, 'milp', with_gap):
            result = super().solve_tree(*args, **kwargs)
        result['metadata'].update(mip_rel_gap_requested=self.relative_gap,
                                  optimality_scope='Both stages; physical checks unchanged')
        return result
