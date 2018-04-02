import numpy as np
import pymc3 as pm
from pymc3.tests.helpers import SeededTest

from l2hmc.l2hmc import L2HMC


class TestL2HMC(SeededTest):
    def test_samples(self):
        chains = 1
        with pm.Model():
            x = pm.Normal('x', 0, 1)
            y = pm.Normal('y', x, 1)

            start, step = pm.init_nuts(chains=chains)

            l2hmc_step = L2HMC(potential=step.potential)
            l2hmc_trace = pm.sample(2000, step=l2hmc_step, start=start, chains=chains)

        assert np.abs(l2hmc_trace['x'].mean()) < 0.02
        assert np.abs(l2hmc_trace['x'].std() - 1) < 0.02
        assert np.abs(l2hmc_trace['y'].mean()) < 0.05
