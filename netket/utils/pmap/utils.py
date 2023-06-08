from functools import partial
from typing import Optional

import jax

from flax.jax_utils import replicate
from jax.sharding import PositionalSharding

from netket.utils.types import SeedT, PRNGKeyT
from ..seed import random_seed

sharding = PositionalSharding(jax.devices())

def split_key(key) -> PRNGKeyT:
    """
    Split a key across MPI nodes in the communicator.
    Only the input key on the root process matters.

    Arguments:
        key: The key to split. Only considered the one on the root process.
        root: (default=0) The root rank from which to take the input key.
        comm: (default=MPI.COMM_WORLD) The MPI communicator.

    Returns:
        A PRNGKey depending on rank number and key.
    """
    #keys = jax.random.split(key, len(jax.devices()))
    #return scatter(keys)
    return key

def PRNGKey(
    seed: Optional[SeedT] = None) -> PRNGKeyT:
    """
    Initialises a PRNGKey using an optional starting seed.
    The same seed will be distributed to all processes.
    """
    if seed is None:
        key = jax.random.PRNGKey(random_seed())
    elif isinstance(seed, int):
        key = jax.random.PRNGKey(seed)
    else:
        key = seed

    key = jax.device_put(key, sharding.replicate(axis=0, keepdims=True))
    return key

def broadcast(x, axis=0):
    return jax.device_put(x, sharding.replicate(axis=axis, keepdims=True))

def _scatter_array(v, axis=0):
    sharding_shape = tuple(1 if i!=axis else -1 for i in range(v.ndim))
    res = jax.device_put(v, sharding.reshape(sharding_shape))
    return res

def scatter(x, axis=0):
    return jax.tree_map(partial(_scatter_array, axis=axis), x)

def global_size(x):
    return x

