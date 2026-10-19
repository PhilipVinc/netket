import os
import re
import signal
import threading
import time
import warnings
from pathlib import Path

import jax
import numpy as np
from flax import serialization
from jax.experimental import multihost_utils

from netket._src import distributed
from netket._src.callbacks.base import StopRun
from netket._src.driver.checkpointer import (
    AbstractCheckpointer,
    Checkpoint,
    _align_keys,
    checkpoint_state,
    flatten_state,
    group_state,
)

_FILE_NAME = re.compile(r"checkpoint_(\d+)\.msgpack")


class Checkpointer(AbstractCheckpointer):
    """
    Saves a driver run to a directory, so that running the same script again
    resumes the run from its last checkpoint, e.g. after a crash or a preemption.

    Pass it to :meth:`~netket.driver.AbstractDriver.run`::

        checkpointer = nk.checkpoint.Checkpointer("run/checkpoints", save_every=100)
        driver.run(5000, out="run/log", checkpointer=checkpointer)

    It saves the run every ``save_every`` steps and where the run ends, and keeps
    the ``keep`` latest checkpoints. A checkpoint holds the driver, its variational
    state and its persistent callbacks (see
    :func:`~netket.checkpoint.checkpoint_state`), in a file
    ``checkpoint_<step>.msgpack`` that process 0 writes and replaces atomically.

    When the process receives one of ``signals`` (by default ``SIGTERM``, which
    Slurm sends before it stops a job), the run saves and stops at the end of its
    current step. Under ``jax.distributed``, JAX handles ``SIGTERM`` itself, and
    the run stops at the step at which all processes reach JAX's preemption sync
    point. Once told to stop, the checkpointer also stops the later runs it is
    given, after their first step.

    With several processes, every process must pass it to ``run`` (and call
    :meth:`latest_step` and :meth:`read`), and ``directory`` must be on a file
    system that they share. A run is resumed on as many devices as it was saved
    from.

    Args:
        directory: The directory of the checkpoints, created at the first save.
        save_every: Save every this many steps (default 100).
        keep: How many of the latest checkpoints to keep (default 1).
        signals: The signals that make the run save and stop (default
            ``(signal.SIGTERM,)``); ``()`` for none.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        save_every: int = 100,
        keep: int = 1,
        signals=(signal.SIGTERM,),
    ):
        if save_every < 1 or keep < 1:
            raise ValueError("save_every and keep must be positive integers.")
        self.directory = Path(directory)
        self.save_every = int(save_every)
        self.keep = int(keep)
        self.signals = tuple(signals)

        self._run = (0, 0)
        self._restored_step = None
        self._saved_step = None
        self._poll_preemption = False
        self._handlers = {}
        # Whether the job was told to stop: kept for the later runs.
        self._stop = False

    def restore(self, driver, *, start_step, target_step):
        self._run = (int(start_step), int(target_step))
        self._restored_step = self._saved_step = None
        self._poll_preemption = (
            signal.SIGTERM in self.signals
            and jax.distributed.is_initialized()
            and jax.config.jax_enable_preemption_service
        )
        # Before reading, so that a signal during a long read is not lost.
        self._install_signal_handlers()

        checkpoint = self.read(target=checkpoint_state(driver))
        if checkpoint is None:
            return None
        if not start_step <= checkpoint.step <= target_step:
            raise ValueError(
                f"The latest checkpoint in '{self.directory}' is of step "
                f"{checkpoint.step}, outside this run from step {start_step} to "
                f"{target_step}: use another directory for another run."
            )
        self._restored_step = checkpoint.step
        return checkpoint.state

    def latest_step(self):
        steps = self._steps() if jax.process_index() == 0 else ()
        return distributed.from_first_process(max(steps, default=None))

    def read(self, step=None, *, target=None):
        if step is None:
            step = self.latest_step()
            if step is None:
                return None
        path = self._path(step)
        if not path.is_file():
            return None
        saved = serialization.msgpack_restore(path.read_bytes())
        state = saved["state"]
        if target is not None:
            state = _like(flatten_state(target), state)
        return Checkpoint(step, group_state(state), saved["metadata"])

    def should_stop(self, step):
        if self._poll_preemption and not self._stop:
            # Every process asks at every step, as JAX requires.
            self._stop = multihost_utils.reached_preemption_sync_point(step)
        if self._stop:
            return StopRun(
                f"Stopped by a signal or a preemption notice at step {step}."
            )
        return None

    def should_save(self, step, *, last):
        due = last or step % self.save_every == 0
        return due and step not in (self._saved_step, self._restored_step)

    def save(self, step, state):
        # On every process: an array no process holds whole is gathered.
        state = _str_keys(_to_host(flatten_state(state)))
        if jax.process_index() == 0:
            from netket import __version__

            metadata = {
                "step": step,
                "start_step": self._run[0],
                "target_step": self._run[1],
                "time": time.time(),
                "netket_version": __version__,
            }
            data = {"metadata": metadata, "state": state}
            self.directory.mkdir(parents=True, exist_ok=True)
            _write(self._path(step), serialization.msgpack_serialize(data))
            older = sorted(s for s in self._steps() if s < step)
            for old in older[: max(len(older) - self.keep + 1, 0)]:
                self._path(old).unlink(missing_ok=True)
        self._saved_step = step

    def close(self, cause=None):
        self._restore_signal_handlers()

    def _path(self, step: int) -> Path:
        return self.directory / f"checkpoint_{step}.msgpack"

    def _steps(self) -> list[int]:
        if not self.directory.is_dir():
            return []
        matches = (_FILE_NAME.fullmatch(p.name) for p in self.directory.iterdir())
        return [int(m.group(1)) for m in matches if m is not None]

    def _install_signal_handlers(self):
        self._restore_signal_handlers()
        if threading.current_thread() is not threading.main_thread():
            if self.signals:
                warnings.warn(
                    "The Checkpointer can handle signals only in the main thread.",
                    stacklevel=3,
                )
            return
        for signum in self.signals:
            if signum == signal.SIGTERM and self._poll_preemption:
                continue  # JAX handles it: see should_stop
            self._handlers[signum] = signal.signal(signum, self._on_signal)

    def _on_signal(self, signum, frame):
        # Only a flag: the run stops at the end of its current step.
        self._stop = True

    def _restore_signal_handlers(self):
        for signum, handler in self._handlers.items():
            if handler is not None:
                signal.signal(signum, handler)
        self._handlers = {}


def _to_host(tree):
    """``tree`` with numpy arrays instead of jax arrays, on every process."""

    def to_host(x):
        if not isinstance(x, jax.Array):
            return x
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x = jax.random.key_data(x)
        if x.is_fully_addressable:
            return np.asarray(x)
        if x.is_fully_replicated:
            return np.asarray(x.addressable_data(0))
        # A collective, in the same order on every process.
        return np.asarray(multihost_utils.process_allgather(x, tiled=True))

    return jax.tree.map(to_host, tree)


def _str_keys(tree):
    """``tree`` with string dict keys, which msgpack reads back."""
    if isinstance(tree, dict):
        return {str(key): _str_keys(value) for key, value in tree.items()}
    return tree


def _like(target, state):
    """``state``, as read, with the structure, dtypes and sharding of ``target``."""
    state = _align_keys(state, target)
    if jax.tree.structure(state) != jax.tree.structure(target):
        raise ValueError(
            "The checkpoint does not have the structure of the driver: was it "
            "saved from another driver, model or set of callbacks?"
        )
    return jax.tree.map(_leaf_like, target, state)


def _leaf_like(target, value):
    if isinstance(target, jax.Array):
        if jax.dtypes.issubdtype(target.dtype, jax.dtypes.prng_key):
            data = _array_like(jax.random.key_data(target), value)
            return jax.random.wrap_key_data(data, impl=jax.random.key_impl(target))
        return _array_like(target, value)
    if isinstance(target, np.ndarray):
        return np.asarray(value, dtype=target.dtype)
    if isinstance(target, np.generic):
        return target.dtype.type(value)
    return value


def _array_like(target, value):
    value = np.asarray(value, dtype=target.dtype)
    if value.shape != target.shape:
        raise ValueError(
            f"The checkpoint has an array of shape {value.shape} where the driver "
            f"has one of shape {target.shape}."
        )
    return jax.make_array_from_callback(
        target.shape, target.sharding, lambda index: value[index]
    )


def _write(path: Path, data: bytes):
    """Writes ``data`` to ``path``: a reader sees the old file or the new one."""
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
