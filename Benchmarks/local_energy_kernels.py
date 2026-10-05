# Copyright 2026 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compares the local-energy kernels of jax operators that skip the zero
connected elements against the padded kernel.

Kernels and operators (each is benchmarked only if the installed NetKet has it):

  padded      `local_value_kernel_jax` (`_chunked`): log psi on all n * max_conn_size
              padded connected configurations.
  flat        `local_value_kernel_jax_flattened`: per device, only the nonzero
              off-diagonal elements, in a data-dependent number of chunks.
  compact     `local_value_kernel_jax_compact` with `CompactConnOperator`: every
              sample packed into a static bound on its nonzero off-diagonal elements.
  spinex      `SpinExchangeOperator`, which only generates the swaps of the
              anti-parallel bonds (padded kernel unless combined with flat/compact).

Spin samples come from a classical antiferromagnet whose fraction of anti-parallel
nearest-neighbour bonds matches the quantum ground state (chain 0.795, square 0.72,
triangular 0.62), because the cost of the kernels only depends on which bonds are
anti-parallel. Hubbard and Ising samples are uniformly random.

Usage:
    python local_energy_kernels.py [--size cpu|gpu] [--models rbm,vit] [--systems ...]

Prints one line per (system, model, kernel) with the median time per batch of
samples, the speedup over the padded kernel with the same chunking on the first
operator of the system, the relative error, and the number
of recompilations over the fresh batches.
"""

import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn

import netket as nk
from netket.vqs.mc import kernels

nkx_op = nk.experimental.operator

HAS_FLAT = hasattr(kernels, "local_value_kernel_jax_flattened")
HAS_COMPACT = hasattr(kernels, "local_value_kernel_jax_compact")
HAS_SPINEX = hasattr(nkx_op, "SpinExchangeOperator")

# Anti-parallel nearest-neighbour fraction of the quantum ground state, from its
# bond energy: chain 1/4 - ln 2, square E/N = -0.6694, triangular E/N = -0.5497.
P_ANTI = {"chain": 0.795, "square": 0.72, "triangular": 0.62}

SIZES = {
    "cpu": {
        "chain": 64,
        "square": 8,
        "triangular": 6,
        "hubbard": 4,
        "ising": 64,
        "n_samples": 512,
        "chunk_size": 4096,
    },
    "gpu": {
        "chain": 256,
        "square": 16,
        "triangular": 12,
        "hubbard": 6,
        "ising": 256,
        "n_samples": 1024,
        "chunk_size": 65536,
        # the unchunked ViT does not fit in memory
        "vit_chunk_size": 16384,
    },
}


## Models


class ViT(nn.Module):
    """A small vision transformer on 1D patches of the flattened lattice."""

    patch_size: int = 4
    features: int = 32
    n_heads: int = 4
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        n, N = x.shape
        x = x.reshape(n, N // self.patch_size, self.patch_size)
        x = nn.Dense(self.features, param_dtype=float)(x)
        pos = self.param("pos", nn.initializers.normal(0.02), x.shape[1:], float)
        x = x + pos
        for _ in range(self.n_layers):
            y = nn.LayerNorm(param_dtype=float)(x)
            x = x + nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, param_dtype=float
            )(y)
            y = nn.LayerNorm(param_dtype=float)(x)
            y = nn.Dense(2 * self.features, param_dtype=float)(y)
            x = x + nn.Dense(self.features, param_dtype=float)(nn.gelu(y))
        x = jnp.sum(x, axis=1)
        re, im = jnp.split(nn.Dense(2, param_dtype=float)(x), 2, axis=-1)
        return (re + 1j * im)[..., 0]


def make_model(name, N):
    if name == "rbm":
        return nk.models.RBM(alpha=4, param_dtype=complex)
    elif name == "vit":
        patch = 4 if N % 4 == 0 else 2 if N % 2 == 0 else 1
        return ViT(patch_size=patch)
    raise ValueError(name)


## Samples


def classical_af_samples(graph, n_samples, p_anti, seed, n_sweeps=200):
    """Samples of an Ising antiferromagnet at zero magnetisation, at the
    temperature giving a fraction `p_anti` of anti-parallel bonds."""
    edges = np.asarray(graph.edges())
    N = graph.n_nodes
    nbrs = [[] for _ in range(N)]
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    nbrs = np.array(nbrs)  # regular lattices

    def run(beta, n, rng):
        s = np.ones((n, N))
        s[:, : N // 2] = -1
        s = rng.permuted(s, axis=1)
        rows = np.arange(n)
        for _ in range(n_sweeps * N // 2):
            i = rng.integers(N, size=n)
            j = rng.integers(N, size=n)
            si, sj = s[rows, i], s[rows, j]
            # energy of J * sum s_i s_j (J = 1) before and after swapping i and j
            hi = (s[rows[:, None], nbrs[i]]).sum(1)
            hj = (s[rows[:, None], nbrs[j]]).sum(1)
            adjacent = (nbrs[i] == j[:, None]).any(1)
            dE = (sj - si) * hi + (si - sj) * hj
            # the bond between i and j stays anti-parallel
            dE = np.where(adjacent, dE - (si - sj) ** 2, dE)
            accept = (si != sj) & (rng.random(n) < np.exp(-beta * dE))
            s[rows[accept], i[accept]] = sj[accept]
            s[rows[accept], j[accept]] = si[accept]
        return s

    def frac(s):
        return np.mean(s[:, edges[:, 0]] != s[:, edges[:, 1]])

    rng = np.random.default_rng(seed)
    lo, hi = 0.0, 4.0
    for _ in range(12):
        beta = (lo + hi) / 2
        if frac(run(beta, 64, rng)) < p_anti:
            lo = beta
        else:
            hi = beta
    return run((lo + hi) / 2, n_samples, rng)


def make_system(name, size, n_samples, n_batches):
    """Returns the hilbert space, the operators to compare (name -> operator,
    bound or None) and a list of sample batches."""
    L = SIZES[size][name]
    ops = {}
    if name in P_ANTI:
        if name == "chain":
            g = nk.graph.Chain(L, pbc=True)
        elif name == "square":
            g = nk.graph.Square(L, pbc=True)
        else:
            g = nk.graph.Triangular([L, L], pbc=True)
        hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
        H = nk.operator.Heisenberg(hi, g).to_jax_operator()
        # provable bounds: every bond on bipartite lattices, 2N on the triangular one
        bound = 2 * g.n_nodes if name == "triangular" else g.n_edges
        ops["LocalOperatorJax"] = (H, bound)
        if HAS_SPINEX:
            Hs = nkx_op.SpinExchangeOperator(hi, g)
            ops["SpinExchangeOperator"] = (Hs, Hs.max_offdiag_conn_size)
        batches = [
            classical_af_samples(g, n_samples, P_ANTI[name], seed)
            for seed in range(n_batches)
        ]
    elif name == "hubbard":
        g = nk.graph.Square(L, pbc=True)
        n_f = g.n_nodes // 2
        hi = nk.hilbert.SpinOrbitalFermions(
            g.n_nodes, s=1 / 2, n_fermions_per_spin=(n_f, n_f)
        )
        c, cdag = nk.operator.fermion.destroy, nk.operator.fermion.create
        nc = nk.operator.fermion.number
        H = 0.0
        for sz in (-1, 1):
            for i, j in g.edges():
                H -= cdag(hi, i, sz) @ c(hi, j, sz) + cdag(hi, j, sz) @ c(hi, i, sz)
        for i in g.nodes():
            H += 4.0 * nc(hi, i, -1) @ nc(hi, i, 1)
        # per spin, at most one of the two hoppings of a bond is nonzero
        ops["FermionOperator2ndJax"] = (H.to_jax_operator(), 2 * g.n_edges)
        ops["FermiHubbardJax"] = (nk.operator.FermiHubbardJax(hi, g, U=4.0), None)
        batches = [
            np.asarray(hi.random_state(jax.random.key(seed), n_samples))
            for seed in range(n_batches)
        ]
    elif name == "ising":
        g = nk.graph.Chain(L, pbc=True)
        hi = nk.hilbert.Spin(0.5, g.n_nodes)
        ops["IsingJax"] = (nk.operator.IsingJax(hi, g, h=1.0), g.n_nodes)
        batches = [
            np.asarray(hi.random_state(jax.random.key(seed), n_samples))
            for seed in range(n_batches)
        ]
    else:
        raise ValueError(name)
    return hi, ops, [jnp.asarray(b, dtype=jnp.int8) for b in batches]


## Kernels


def kernel_variants(chunk_size):
    """name -> (kernel(logpsi, pars, σ, O), whether the operator must be wrapped in
    a CompactConnOperator)."""
    out = {}
    out["padded"] = (kernels.local_value_kernel_jax, False)
    out["padded/chunk"] = (
        lambda *a: kernels.local_value_kernel_jax_chunked(*a, chunk_size=chunk_size),
        False,
    )
    if HAS_FLAT:
        out["flat"] = (kernels.local_value_kernel_jax_flattened, False)
        out["flat/chunk"] = (
            lambda *a: kernels.local_value_kernel_jax_flattened(
                *a, chunk_size=chunk_size
            ),
            False,
        )
    if HAS_COMPACT:
        out["compact"] = (kernels.local_value_kernel_jax_compact, True)
        out["compact/chunk"] = (
            lambda *a: kernels.local_value_kernel_jax_compact(
                *a, chunk_size=chunk_size
            ),
            True,
        )
    return out


@jax.jit
def n_offdiag(O, σ):
    xp, mels = O.get_conn_padded(σ)
    is_diag = jnp.all(xp == σ[:, None, :], axis=-1)
    return jnp.sum((mels != 0) & ~is_diag, axis=-1)


def bench(apply_fun, variables, O, batches, kernel, n_repeats):
    n_traces = 0

    def logpsi(v, x):
        nonlocal n_traces
        n_traces += 1
        return apply_fun(v, x)

    f = jax.jit(lambda v, σ, O: kernel(logpsi, v, σ, O))

    t0 = time.perf_counter()
    out = [f(variables, batches[0], O).block_until_ready()]
    t_first = time.perf_counter() - t0
    traces_first = n_traces

    times = []
    for σ in batches[1:]:
        out.append(f(variables, σ, O).block_until_ready())
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            f(variables, σ, O).block_until_ready()
            times.append(time.perf_counter() - t0)
    recompiles = n_traces != traces_first
    return np.median(times), t_first, recompiles, np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="cpu", choices=list(SIZES))
    p.add_argument("--models", default="rbm,vit")
    p.add_argument("--systems", default="chain,square,triangular,hubbard,ising")
    p.add_argument("--kernels", default=None)
    p.add_argument("--n-batches", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=3)
    args = p.parse_args()

    n_samples = SIZES[args.size]["n_samples"]
    chunk_size = SIZES[args.size]["chunk_size"]

    def model_variants(model_name):
        cs = SIZES[args.size].get(f"{model_name}_chunk_size")
        variants = kernel_variants(cs or chunk_size)
        if cs is not None:
            variants = {k: v for k, v in variants.items() if k.endswith("/chunk")}
        if args.kernels is not None:
            variants = {
                k: v for k, v in variants.items() if k in args.kernels.split(",")
            }
        return variants

    print(f"# devices: {jax.devices()}  netket: {nk.__version__}")
    print(f"# {SIZES[args.size]}")
    print(
        "| system | model | operator | M | offdiag/sample | kernel "
        "| ms/batch | vs padded | rel err (system / operator) | first call (s) "
        "| recompiles |"
    )
    print("|---|---|---|---:|---:|---|---:|---:|---:|---:|---|")
    for system in args.systems.split(","):
        hi, ops, batches = make_system(system, args.size, n_samples, args.n_batches + 1)
        for model_name in args.models.split(","):
            model = make_model(model_name, hi.size)
            variants = model_variants(model_name)
            variables = model.init(jax.random.key(0), batches[0][:2])
            apply_fun = model.apply
            # speedups are relative to the padded kernel on the first operator,
            # with the same chunking
            t_ref = {}
            ref = None
            for op_name, (O, bound) in ops.items():
                n_off = float(jnp.mean(n_offdiag(O, batches[0])))
                op_ref = None
                for kname, (kernel, compact) in variants.items():
                    O_k = O
                    if compact:
                        if bound is None:
                            continue
                        O_k = nkx_op.CompactConnOperator(O, bound)
                    t, t_first, recompiles, out = bench(
                        apply_fun, variables, O_k, batches, kernel, args.n_repeats
                    )
                    mode = kname.partition("/")[2]
                    t_ref.setdefault(mode, t)
                    if op_ref is None:
                        op_ref = out
                    if ref is None:
                        ref = out
                    # relative to the first operator (the same Hamiltonian), and to
                    # the padded kernel on this operator
                    err = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
                    err_op = float(
                        np.max(np.abs(out - op_ref)) / np.max(np.abs(op_ref))
                    )
                    print(
                        f"| {system} N={hi.size} | {model_name} | {op_name} "
                        f"| {O_k.max_conn_size} | {n_off:.1f} | {kname} "
                        f"| {1e3 * t:.2f} | {t_ref[mode] / t:.2f}x | {err:.1e} / {err_op:.1e} "
                        f"| {t_first:.1f} | {'yes' if recompiles else 'no'} |",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
