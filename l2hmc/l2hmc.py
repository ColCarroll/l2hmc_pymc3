from collections import namedtuple

import numpy as np
from pymc3 import HamiltonianMC
from pymc3.step_methods.hmc.integration import IntegrationError
from pymc3.step_methods.hmc.base_hmc import DivergenceInfo, HMCStepData

from .integration import L2HMCLeapfrogIntegrator


def _zeros_func(q, p, t):
    return np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)

def default_aux_functions():
    """Returns auxilliary functions that are all identically 0"""
    return _zeros_func, _zeros_func


class L2HMC(HamiltonianMC):
    def __init__(self, q_func=None, p_func=None, *args, **kwargs):
        if (q_func is None) and (p_func is None):
            q_func, p_func = default_aux_functions()
        super(L2HMC, self).__init__(*args, **kwargs)
        self.integrator = L2HMCLeapfrogIntegrator(
            self.potential, self._logp_dlogp_func, q_func=q_func, p_func=p_func)

    def _hamiltonian_step(self, start, p0, step_size):
        path_length = np.random.rand() * self.path_length
        n_steps = max(1, int(path_length / step_size))

        energy_change = -np.inf
        state = start
        div_info = None
        try:
            for _ in range(n_steps):
                state = self.integrator.step(step_size, state)
        except IntegrationError as e:
            div_info = DivergenceInfo('Divergence encountered.', e, state)
        else:
            if not np.isfinite(state.energy):
                div_info = DivergenceInfo('Divergence encountered, bad energy.', None, state)
            energy_change = start.energy - state.energy
            if np.abs(energy_change) > self.Emax:
                div_info = DivergenceInfo('Divergence encountered, large integration error.',
                                          None, state)

        accept_stat = min(1, np.exp(energy_change + state.log_jac))

        if div_info is not None or np.random.rand() >= accept_stat:
            end = start
            accepted = False
        else:
            end = state
            accepted = True

        stats = {
            'path_length': path_length,
            'n_steps': n_steps,
            'accept': accept_stat,
            'energy_error': energy_change,
            'energy': state.energy,
            'accepted': accepted,
            # 'log_jacobian': state.log_jac,  # TODO: backends/ndarray.py to allow this
        }
        return HMCStepData(end, accept_stat, div_info, stats)
