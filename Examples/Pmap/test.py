import os
os.environ["MPI4JAX_NO_WARN_JAX_VERSION"] = "1"
os.environ["NETKET_EXPERIMENTAL_PMAP"] = "8"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=8"

import netket as nk
import jax
print(jax.devices())

import jax.numpy as jnp
from functools import partial

import functools
import numpy as np
import jax

from jax import lax, random, numpy as jnp

import flax
from flax import struct, traverse_util, linen as nn
from flax.linen import spmd # Flax Linen SPMD.
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints

import optax # Optax for common losses and optimizers. 

from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# Create a Sharding object to distribute a value across devices:
sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
sharding = PositionalSharding(jax.devices())

# Create an array of random values:
x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, sharding.reshape(4, 2))
jax.debug.visualize_array_sharding(y)

import jax
x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))


hi = nk.hilbert.Spin(0.5, 3)
ma = nk.models.RBM()
sa = nk.sampler.MetropolisLocal(hi)

k = jax.random.PRNGKey(3)
x = hi.all_states()
xb = hi.random_state(k, (8, 30))
xb = jax.device_put(xb, sharding.reshape(-1, 1, 1))

pars = ma.init(jax.random.PRNGKey(1), xb)
parsr = jax.device_put(pars, sharding.replicate(axis=0, keepdims=True))

sa_state = sa.init_state(ma, parsr)









