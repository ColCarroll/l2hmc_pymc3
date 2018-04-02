import numpy as np
import numpy.testing as npt
import pymc3 as pm

from l2hmc.integration import random_binary_array, L2HMCLeapfrogIntegrator
from l2hmc.l2hmc import default_aux_functions


def test_random_binary_array():
    b = random_binary_array(101)
    assert b.sum() == 50
    assert b.dtype == np.dtype('bool')
    # Make sure it is actually shuffled.  Will fail _super_ occasionally
    assert 0 < b[:50].sum() < 50


def test_random_binary_array_update():
    b = random_binary_array(100)
    x = np.random.rand(len(b))
    x_new = x.copy()

    x_new[b] = (x_new + 1)[b]
    x_new[~b] = (x_new + 1)[~b]
    npt.assert_array_almost_equal(x_new, x + 1)

def test_l2hmc_matches_leapfrog():
    with pm.Model():
        x = pm.Normal('x', 0, 1)
        y = pm.Normal('y', x, 1)
        step = pm.NUTS()

    q_func, p_func = default_aux_functions()
    l2hmc_integrator = L2HMCLeapfrogIntegrator(step.potential, step._logp_dlogp_func, q_func=q_func, p_func=p_func)
    hmc_integrator = pm.step_methods.hmc.integration.CpuLeapfrogIntegrator(step.potential, step._logp_dlogp_func)

    points = []
    p0 = step.potential.random()
    for integrator in (l2hmc_integrator, hmc_integrator):
        point = {'x': np.array([1.]), 'y': np.array([1.])}
        integrator._logp_dlogp_func.set_extra_values(point)
        q0 = integrator._logp_dlogp_func.dict_to_array(point)
        state = integrator.compute_state(q0, p0)
        points.append(integrator._step(0.1, state))

    l2hmc_state, hmc_state = points
    npt.assert_array_almost_equal(l2hmc_state.q, hmc_state.q)
    npt.assert_array_almost_equal(l2hmc_state.p, hmc_state.p)
    npt.assert_array_almost_equal(l2hmc_state.v, hmc_state.v)
    npt.assert_array_almost_equal(l2hmc_state.q_grad, hmc_state.q_grad)
    assert l2hmc_state.energy == hmc_state.energy
