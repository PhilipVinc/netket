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

sharding = nk.utils.pmap.sharding

hi = nk.hilbert.Spin(0.5, 3)
ma = nk.models.RBM()
sa = nk.sampler.MetropolisLocal(hi)
esa = nk.sampler.ExactSampler(hi)

k = jax.random.PRNGKey(3)
x = hi.all_states()
xb = hi.random_state(k, (8, 30))

pars = ma.init(jax.random.PRNGKey(1), xb)
parsr = jax.device_put(pars, sharding.replicate(axis=0, keepdims=True))

esa_state = esa.init_state(ma, pars)
esa_state = esa.reset(ma, pars, state=esa_state)
esamples, esa_state2 = esa.sample(ma, pars, state=esa_state)

sa_state = sa.init_state(ma, pars)
sa_state = sa.reset(ma, pars, state=sa_state)
samples, sa_state2 = sa.sample(ma, pars, state=sa_state)

ha = nk.operator.Ising(hi, nk.graph.Chain(hi.size), h=1.0)

vs = nk.vqs.MCState(sa, ma)

vs.samples
r=vs.expect(ha)
print(r)
