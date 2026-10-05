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
Run-to-run repeatability, accuracy and throughput of the flattened local-energy
kernel, against the padded kernel and against the first version of the
flattened kernel, which reduced the terms of every sample with
`jax.ops.segment_sum` (a scatter-add with several updates per sample).

Kernels:

  padded       `local_value_kernel_jax` (`_chunked`).
  flat         `local_value_kernel_jax_flattened`: writes log psi back to the
               padded layout (no overlapping updates) and sums like `padded`.
  flat-segsum  the first version of the flattened kernel (forward pass only,
               copied below), reducing with `segment_sum`.

For every (system, model, kernel) it evaluates the kernel `--n-calls` times on
the same inputs and counts the distinct results (bitwise), and records a hash
of the result. Running the script in two fresh processes with the same
`--inputs` directory (the first run saves the samples and the variables, the
next ones load them) and comparing the `--hashes` files with `--compare`
checks the repeatability across processes.

Usage:
    python local_energy_determinism.py --size gpu --inputs DIR --hashes run1.json
    python local_energy_determinism.py --size gpu --inputs DIR --hashes run2.json
    python local_energy_determinism.py --compare run1.json run2.json
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import flax

import netket as nk
import netket.jax as nkjax
from netket.jax.sharding import sharding_decorator
from netket.vqs.mc import kernels

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_energy_kernels import SIZES, make_model, make_system  # noqa: E402


## The first version of the flattened kernel (forward pass only).


def segment_sum_kernel(logpsi, pars, σ, O, *, chunk_size=None):
    kernel = nkjax.HashablePartial(_segment_sum_kernel, logpsi, chunk_size=chunk_size)
    return sharding_decorator(
        kernel,
        sharded_args_tree=(False, True, False),
        pvary_args_tree=(True, False, False),
    )(pars, σ, O)


def _segment_sum_kernel(logpsi, pars, σ, O, *, chunk_size):
    n_samples, N = σ.shape
    σp, mels = O.get_conn_padded(σ)
    max_conn_size = mels.shape[-1]
    n_conns = n_samples * max_conn_size

    is_diag = jnp.all(σp == jnp.expand_dims(σ, -2), axis=-1)
    mels_diag = jnp.sum(jnp.where(is_diag, mels, 0), axis=-1)
    mask = (mels != 0) & ~is_diag

    if chunk_size is None:
        chunk_size = -(-n_conns // 16)
    chunk_size = max(1, min(chunk_size, n_conns))
    n_chunks_max = -(-n_conns // chunk_size)

    n_nonzero = mask.sum()
    (idx,) = jnp.nonzero(
        mask.reshape(-1), size=n_chunks_max * chunk_size, fill_value=n_conns - 1
    )
    logpsi_σ = nkjax.apply_chunked(
        logpsi, in_axes=(None, 0), chunk_size=chunk_size, axis_0_is_sharded=False
    )(pars, σ)
    σp = σp.reshape(-1, N)
    mels = mels.reshape(-1)

    def body(c, acc):
        i = jax.lax.dynamic_slice_in_dim(idx, c * chunk_size, chunk_size)
        sample = i // max_conn_size
        terms = mels[i] * jnp.exp(logpsi(pars, σp[i]) - logpsi_σ[sample])
        is_valid = c * chunk_size + jnp.arange(chunk_size) < n_nonzero
        terms = jnp.where(is_valid, terms, 0)
        return acc + jax.ops.segment_sum(
            terms, sample, num_segments=n_samples, indices_are_sorted=True
        )

    dtype = jnp.result_type(mels.dtype, logpsi_σ.dtype)
    n_chunks = (n_nonzero + chunk_size - 1) // chunk_size
    acc = jax.lax.fori_loop(0, n_chunks, body, jnp.zeros_like(logpsi_σ, dtype=dtype))
    return mels_diag + acc


def kernel_variants(chunk_size, unchunked):
    out = {}
    if unchunked:
        out["padded"] = kernels.local_value_kernel_jax
        out["flat-segsum"] = segment_sum_kernel
        out["flat"] = kernels.local_value_kernel_jax_flattened
    out["padded/chunk"] = lambda *a: kernels.local_value_kernel_jax_chunked(
        *a, chunk_size=chunk_size
    )
    out["flat-segsum/chunk"] = lambda *a: segment_sum_kernel(*a, chunk_size=chunk_size)
    out["flat/chunk"] = lambda *a: kernels.local_value_kernel_jax_flattened(
        *a, chunk_size=chunk_size
    )
    return out


## Saved inputs


def load_or_save_inputs(path, σ, variables):
    """The first run saves the inputs, the next ones load them."""
    if os.path.exists(path):
        data = np.load(path)
        σ = jnp.asarray(data["samples"])
        variables = flax.serialization.msgpack_restore(data["variables"].tobytes())
        return σ, jax.tree.map(jnp.asarray, variables), "loaded"
    np.savez(
        path,
        samples=np.asarray(σ),
        variables=np.frombuffer(
            flax.serialization.msgpack_serialize(jax.device_get(variables)),
            dtype=np.uint8,
        ),
    )
    return σ, variables, "saved"


def digest(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:16]


def run(args):
    sizes = SIZES[args.size]
    n_samples, chunk_size = sizes["n_samples"], sizes["chunk_size"]
    os.makedirs(args.inputs, exist_ok=True)

    print(f"# devices: {jax.devices()}  jax {jax.__version__}  netket {nk.__version__}")
    print(f"# XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}  {sizes}")
    print(
        "| system | model | kernel | ms/batch | vs padded | max rel. dev. from padded "
        "| distinct results / calls | hash |"
    )
    print("|---|---|---|---:|---:|---:|---:|---|")
    hashes = {}
    for system in args.systems.split(","):
        hi, ops, batches = make_system(system, args.size, n_samples, 1)
        O = next(iter(ops.values()))[0]
        for model_name in args.models.split(","):
            model = make_model(model_name, hi.size)
            variables = model.init(jax.random.key(0), batches[0][:2])
            σ, variables, how = load_or_save_inputs(
                os.path.join(args.inputs, f"{system}-{model_name}.npz"),
                batches[0],
                variables,
            )
            cs = sizes.get(f"{model_name}_chunk_size")
            variants = kernel_variants(cs or chunk_size, unchunked=cs is None)
            t_ref, ref = {}, {}
            for kname, kernel in variants.items():
                f = jax.jit(lambda v, σ, O, kernel=kernel: kernel(model.apply, v, σ, O))
                out = f(variables, σ, O).block_until_ready()
                results = {digest(out)}
                times = []
                for _ in range(args.n_calls):
                    t0 = time.perf_counter()
                    out_k = f(variables, σ, O).block_until_ready()
                    times.append(time.perf_counter() - t0)
                    results.add(digest(out_k))
                t = np.median(times)
                mode = kname.partition("/")[2]
                t_ref.setdefault(mode, t)
                ref.setdefault(mode, np.asarray(out))
                dev = float(
                    np.max(np.abs(np.asarray(out) - ref[mode]))
                    / np.max(np.abs(ref[mode]))
                )
                key = f"{system}|{model_name}|{kname}"
                hashes[key] = digest(out)
                print(
                    f"| {system} N={hi.size} | {model_name} | {kname} "
                    f"| {1e3 * t:.2f} | {t_ref[mode] / t:.2f}x | {dev:.1e} "
                    f"| {len(results)} / {args.n_calls + 1} | {hashes[key]} ({how}) |",
                    flush=True,
                )
    if args.hashes is not None:
        with open(args.hashes, "w") as f:
            json.dump(hashes, f, indent=1)


def compare(files):
    runs = [json.load(open(f)) for f in files]
    keys = sorted(set().union(*runs))
    print("| system | model | kernel | identical in all processes |")
    print("|---|---|---|---|")
    for key in keys:
        values = [r.get(key) for r in runs]
        same = len(set(values)) == 1 and None not in values
        print(f"| {' | '.join(key.split('|'))} | {'yes' if same else 'NO'} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="cpu", choices=list(SIZES))
    p.add_argument("--models", default="rbm,vit")
    p.add_argument("--systems", default="chain,square,triangular,hubbard,ising")
    p.add_argument("--n-calls", type=int, default=20)
    p.add_argument("--inputs", default="local_energy_determinism_inputs")
    p.add_argument("--hashes", default=None)
    p.add_argument("--compare", nargs="+", default=None)
    args = p.parse_args()
    if args.compare is not None:
        compare(args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
