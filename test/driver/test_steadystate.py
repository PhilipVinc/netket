import pytest
from pytest import raises

import numpy as np
import netket as nk

from .. import common

pytestmark = common.skipif_mpi

SEED = 214748364


def _setup_system():
    L = 3

    hi = nk.hilbert.Spin(s=0.5) ** L

    ha = nk.operator.LocalOperator(hi)
    j_ops = []
    for i in range(L):
        ha += (0.3 / 2.0) * nk.operator.spin.sigmax(hi, i)
        ha += (
            (2.0 / 4.0)
            * nk.operator.spin.sigmaz(hi, i)
            * nk.operator.spin.sigmaz(hi, (i + 1) % L)
        )
        j_ops.append(nk.operator.spin.sigmam(hi, i))

    #  Create the liouvillian
    lind = nk.operator.LocalLiouvillian(ha, j_ops)
    return hi, lind


def _setup_ss(dtype=np.float32, sr=True):
    hi, lind = _setup_system()

    ma = nk.models.NDM()
    # sa = nk.sampler.ExactSampler(hilbert=nk.hilbert.DoubledHilber(hi))

    sa = nk.sampler.MetropolisLocal(hilbert=nk.hilbert.DoubledHilbert(hi))
    sa_obs = nk.sampler.MetropolisLocal(hilbert=hi)

    vs = nk.vqs.MCMixedState(sa, ma, sampler_diag=sa_obs, n_samples=1008, seed=SEED)

    op = nk.optimizer.Sgd(learning_rate=0.05)
    if sr:
        sr_config = nk.optimizer.SR(holomorphic=False)
    else:
        sr_config = None

    driver = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr_config)

    return lind, vs, driver


def _setup_obs(L):
    hi = nk.hilbert.Spin(s=0.5) ** L

    obs_sx = nk.operator.LocalOperator(hi)
    for i in range(L):
        obs_sx += nk.operator.spin.sigmax(hi, i)

    obs = {"SigmaX": obs_sx}
    return obs


####


def test_estimate():
    lind, _, driver = _setup_ss()

    driver.estimate(lind.H @ lind)
    driver.advance(1)
    driver.estimate(lind.H @ lind)


def test_raise_n_iter():
    lind, _, driver = _setup_ss()
    with raises(ValueError):
        driver.run("prova", 12)


def test_no_preconditioner_api():
    lind, vs, driver = _setup_ss()

    driver.preconditioner = None
    assert driver.preconditioner(None, 1) == 1
    assert driver.preconditioner(None, 1, 2) == 1


def test_preconditioner_deprecated_signature():
    lind, vs, driver = _setup_ss()

    sr = driver.preconditioner
    _sr = lambda vstate, grad: sr(vstate, grad)

    with pytest.warns(FutureWarning):
        driver.preconditioner = _sr

    driver.run(1)
