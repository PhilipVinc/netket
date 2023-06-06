

import jax
import jax.numpy as jnp
import jaxlib
from jax.sharding import PositionalSharding

def sum_jax(arr):
    if isinstance(arr, jaxlib.xla_extension.pmap_lib.ShardedDeviceArray):
        return jnp.sum(arr, axis=0), None
    else:
        return jax.lax.psum(arr, 'mpi'), None

def mean_jax(arr):
    if isinstance(arr, jaxlib.xla_extension.pmap_lib.ShardedDeviceArray):
        return jnp.mean(arr, axis=0), None
    else:
        return jax.lax.pmean(arr, 'mpi'), None

def max_jax(arr):
    if isinstance(arr, jaxlib.xla_extension.pmap_lib.ShardedDeviceArray):
        return jnp.max(arr, axis=0), None
    else:
        return jax.lax.pmax(arr, 'mpi'), None

sharding = PositionalSharding(jax.devices())

def make_array_from_callback(input_shape, cb):
    _sharding = sharding.reshape(-1, *tuple(1 for _ in range(len(input_shape) -1) ))
    return jax.make_array_from_callback(input_shape, _sharding, cb)