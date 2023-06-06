import os
os.environ["MPI4JAX_NO_WARN_JAX_VERSION"] = "1"
os.environ["NETKET_EXPERIMENTAL_PMAP"] = "4"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=4"

import netket as nk
import jax
print(jax.devices())

import jax.numpy as jnp
from functools import partial

hi = nk.hilbert.Spin(0.5, 3)
ma = nk.models.RBM()
sa = nk.sampler.MetropolisLocal(hi)

pars = ma.init(jax.random.PRNGKey(1), hi.all_states())
sa_state = sa.init_state(ma, pars)

ha = nk.operator.Ising(hi, nk.graph.Chain(hi.size), h=1.0)

vs = nk.vqs.MCState(sa, ma)

vs.samples
r=vs.expect(ha)
print(r)