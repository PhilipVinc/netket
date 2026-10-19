import os
import signal
import threading

import jax
import numpy as np
import pytest
from jax.experimental import multihost_utils

import netket as nk
from netket.callbacks import AbstractCallback
from netket.checkpoint import Checkpointer
from netket.utils import struct


def _vmc():
    hi = nk.hilbert.Spin(0.5, 4)
    ha = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=1.0)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=4)
    model = nk.models.RBM(alpha=1, param_dtype=float)
    vs = nk.vqs.MCState(sampler, model, n_samples=64, seed=12, sampler_seed=13)
    return nk.driver.VMC(ha, nk.optimizer.Sgd(0.05), variational_state=vs)


def _assert_same(got, want):
    def host(x):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x = jax.random.key_data(x)
        if isinstance(x, jax.Array) and not x.is_fully_addressable:
            x = multihost_utils.process_allgather(x, tiled=True)
        return np.asarray(x)

    for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(host(x), host(y))


class _Boom(Exception):
    pass


class _CrashAt(AbstractCallback):
    _at: int = struct.field(pytree_node=False, serialize=False)

    def __init__(self, at):
        self._at = at

    def on_compute_update_start(self, step, log_data, driver):
        if step == self._at:
            raise _Boom("simulated interruption")


class _SignalAt(AbstractCallback):
    """Sends ``signum`` to process 0 at the end of step ``at``."""

    _at: int = struct.field(pytree_node=False, serialize=False)
    _signum: int = struct.field(pytree_node=False, serialize=False)

    def __init__(self, at, signum):
        self._at = at
        self._signum = signum

    def on_step_end(self, step, log_data, driver):
        if step == self._at and jax.process_index() == 0:
            os.kill(os.getpid(), self._signum)


def _steps(directory):
    return sorted(int(name[11:-8]) for name in os.listdir(directory))


def test_a_crashed_run_resumes_exactly(tmp_path_distributed):
    directory = tmp_path_distributed / "checkpoints"
    reference = _vmc()
    reference.run(8, show_progress=False)

    with pytest.raises(_Boom):
        _vmc().run(
            8,
            callback=_CrashAt(5),
            checkpointer=Checkpointer(directory, save_every=2),
            show_progress=False,
        )
    assert Checkpointer(directory).latest_step() == 4

    resumed = _vmc()
    resumed.run(
        8, checkpointer=Checkpointer(directory, save_every=2), show_progress=False
    )
    assert resumed.step_count == 8
    _assert_same(resumed.state.parameters, reference.state.parameters)
    _assert_same(resumed.state.sampler_state, reference.state.sampler_state)
    if jax.process_index() == 0:
        assert _steps(directory) == [8]


def test_it_saves_periodically_and_at_the_end(tmp_path_distributed):
    directory = tmp_path_distributed / "checkpoints"
    driver = _vmc()
    checkpointer = Checkpointer(directory, save_every=2, keep=10)
    driver.run(5, checkpointer=checkpointer, show_progress=False)
    if jax.process_index() == 0:
        assert _steps(directory) == [2, 4, 5]

    # Read without a driver.
    checkpoint = checkpointer.read()
    assert checkpoint.step == checkpointer.latest_step() == 5
    assert set(checkpoint.state) == {"driver", "variational_state", "callbacks"}
    assert checkpoint.metadata["start_step"] == 0
    assert checkpoint.metadata["target_step"] == 5
    assert checkpointer.read(2).step == 2 and checkpointer.read(3) is None

    # Seed a new run with the parameters of the last checkpoint.
    new = _vmc()
    nk.checkpoint.load_checkpoint_state(
        new, checkpoint.state, only=("variational_state/variables",)
    )
    _assert_same(new.state.parameters, driver.state.parameters)
    assert new.step_count == 0


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGUSR1])
def test_a_signal_saves_and_stops_the_run(tmp_path_distributed, signum):
    if signum == signal.SIGTERM and jax.process_count() > 1:
        pytest.skip("Under jax.distributed, JAX handles SIGTERM.")
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("Signal handlers can be installed only in the main thread.")
    directory = tmp_path_distributed / "checkpoints"
    handler = signal.getsignal(signum)
    checkpointer = Checkpointer(directory, signals=(signum,))
    driver = _vmc()
    driver.run(
        10,
        callback=_SignalAt(2, signum),
        checkpointer=checkpointer,
        show_progress=False,
    )
    # Stopped at the end of step 2, on every process, saved there.
    assert driver.step_count == 3
    assert checkpointer.latest_step() == 3
    assert signal.getsignal(signum) is handler

    # Once told to stop, it stops the later runs after their first step.
    driver.run(10, checkpointer=checkpointer, show_progress=False)
    assert driver.step_count == 4


def test_the_checkpoint_of_another_run_is_refused(tmp_path_distributed):
    directory = tmp_path_distributed / "checkpoints"
    _vmc().run(6, checkpointer=Checkpointer(directory), show_progress=False)
    with pytest.raises(ValueError, match="outside this run"):
        _vmc().run(4, checkpointer=Checkpointer(directory), show_progress=False)
