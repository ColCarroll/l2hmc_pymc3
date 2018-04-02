from collections import namedtuple

import numpy as np
from pymc3.step_methods.hmc.integration import IntegrationError
from scipy import linalg


State = namedtuple("State", 'q, p, v, q_grad, energy, log_jac')


def random_binary_array(total_len):
    """Generate a binary array with half the entries 1"""
    positive = total_len // 2
    arr = np.zeros(total_len, dtype=bool)
    arr[:positive]  = True
    np.random.shuffle(arr)
    return arr


class L2HMCLeapfrogIntegrator(object):
    def __init__(self, potential, logp_dlogp_func, q_func, p_func):
        """Leapfrog integrator using CPU."""
        self._potential = potential
        self._logp_dlogp_func = logp_dlogp_func
        self._dtype = self._logp_dlogp_func.dtype
        self.q_func, self.p_func = q_func, p_func
        if self._potential.dtype != self._dtype:
            raise ValueError("dtypes of potential (%s) and logp function (%s)"
                             "don't match."
                             % (self._potential.dtype, self._dtype))

    def compute_state(self, q, p):
        """Compute Hamiltonian functions using a position and momentum."""
        if q.dtype != self._dtype or p.dtype != self._dtype:
            raise ValueError('Invalid dtype. Must be %s' % self._dtype)
        logp, dlogp = self._logp_dlogp_func(q)
        v = self._potential.velocity(p)
        kinetic = self._potential.energy(p, velocity=v)
        energy = kinetic - logp
        return State(q, p, v, dlogp, energy, 0)

    def step(self, epsilon, state, out=None):
        """Leapfrog integrator step.

        Half a momentum update, full position update, half momentum update.

        Parameters
        ----------
        epsilon: float, > 0
            step scale
        state: State namedtuple,
            current position data
        out: (optional) State namedtuple,
            preallocated arrays to write to in place

        Returns
        -------
        None if `out` is provided, else a State namedtuple
        """
        try:
            return self._step(epsilon, state, out=None)
        except linalg.LinAlgError as err:
            msg = "LinAlgError during leapfrog step."
            raise IntegrationError(msg)
        except ValueError as err:
            # Raised by many scipy.linalg functions
            scipy_msg = "array must not contain infs or nans"
            if len(err.args) > 0 and scipy_msg in err.args[0].lower():
                msg = "Infs or nans in scipy.linalg during leapfrog step."
                raise IntegrationError(msg)
            else:
                raise

    def _step(self, epsilon, state, out=None):
        pot = self._potential

        q, p, v, q_grad, energy, log_jac = state
        if out is None:
            q_new = q.copy()
            p_new = p.copy()
            v_new = np.empty_like(q)
            q_new_grad = np.empty_like(q)
        else:
            q_new, p_new, v_new, q_new_grad, energy = out
            q_new[:] = q
            p_new[:] = p

        dt = 0.5 * epsilon
        m = random_binary_array(len(q))

        # TODO: set this value
        t = 0

        # p is already stored in p_new
        zeta_1 = (q_new, q_grad, t)
        Q_p_1, S_p_1, T_p_1 = self.p_func(*zeta_1)
        first_S_p = dt * S_p_1
        p_new = p_new * np.exp(first_S_p) + dt * (q_grad * np.exp(epsilon * Q_p_1) + T_p_1)

        pot.velocity(p_new, out=v_new)
        # q is already stored in q_new
        zeta_2 = (q_new, v_new, t)
        Q_q_1, S_q_1, T_q_1 = self.q_func(*zeta_2)
        first_S_q = epsilon * S_q_1
        q_new[m] =  (q_new * np.exp(first_S_q) + epsilon * (v_new * np.exp(epsilon * Q_q_1) + T_q_1))[m]

        zeta_3 = (q_new, v_new, t)
        Q_q_2, S_q_2, T_q_2 = self.q_func(*zeta_3)
        second_S_q = epsilon * S_q_2
        q_new[~m] = (q_new * np.exp(second_S_q) + epsilon * (v_new * np.exp(epsilon * Q_q_2) + T_q_2))[~m]

        logp = self._logp_dlogp_func(q_new, q_new_grad)
        zeta_4 = (q_new, q_new_grad, t)
        Q_p_2, S_p_2, T_p_2 = self.p_func(*zeta_4)
        second_S_p = dt * S_p_2

        p_new = p_new * np.exp(second_S_p) + dt * (q_new_grad * np.exp(epsilon * Q_p_2) + T_p_2)

        kinetic = pot.velocity_energy(p_new, v_new)
        energy = kinetic - logp
        log_jac += (first_S_p + second_S_p).sum() + first_S_q[m].sum() + second_S_q[m].sum()

        if out is not None:
            out.energy = energy
            out.log_jac = log_jac
            return
        else:
            return State(q_new, p_new, v_new, q_new_grad, energy, log_jac)