from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from jax.sharding import NamedSharding, PartitionSpec as P

from netket import jax as nkjax
from netket import config
from netket.jax._jacobian.default_mode import JacobianMode
from netket.utils import timing
from netket.utils.types import Array


def _generate_random_orthogonal_vectors(key, num_vectors, dimension):
    """
    Generate num_vectors random orthogonal unit vectors in dimension-dimensional space.
    Uses QR decomposition to ensure orthogonality.

    Args:
        key: PRNG key
        num_vectors: Number of orthogonal vectors to generate (M)
        dimension: Dimension of the parameter space (Npars)

    Returns:
        Array of shape (dimension, num_vectors) where each column is a unit vector
    """
    # Generate random matrix
    random_matrix = jax.random.normal(key, (dimension, num_vectors))
    # QR decomposition to get orthogonal vectors
    Q, _ = jnp.linalg.qr(random_matrix)
    return Q  # shape: (dimension, num_vectors)


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "chunk_size",
        "mode",
        "projection_dim",
    ),
)
def srt_onthefly(
    log_psi,
    local_energies,
    parameters,
    model_state,
    samples,
    *,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: JacobianMode,
    projection_dim: int | None = None,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    chunk_size: int | None = None,
    rng_key: jax.Array | None = None,
):
    N_mc = local_energies.size

    # Split all parameters into real and imaginary parts separately
    parameters_real, rss = nkjax.tree_to_real(parameters)

    # Flatten parameters to get dimension
    from jax.flatten_util import ravel_pytree
    parameters_flat, unravel_fn = ravel_pytree(parameters_real)
    n_params = parameters_flat.size

    # complex: (Nmc) -> (Nmc,2) - splitting real and imaginary output like 2 classes
    # real:    (Nmc) -> (Nmc,)  - no splitting
    def _apply_fn(parameters_real, samples):
        variables = {"params": rss(parameters_real), **model_state}
        log_amp = log_psi(variables, samples)

        if mode == "complex":
            re, im = log_amp.real, log_amp.imag
            return jnp.concatenate(
                (re[:, None], im[:, None]), axis=-1
            )  # shape [N_mc,2]
        else:
            return log_amp.real  # shape [N_mc, ]

    def jvp_f_chunk(parameters, vector, samples):
        r"""
        Creates the jvp of the function `_apply_fn` with respect to the parameters.
        This jvp is then evaluated in chunks of `chunk_size` samples.
        """
        f = lambda params: _apply_fn(params, samples)
        _, acc = jax.jvp(f, (parameters,), (vector,))
        return acc

    # compute rhs of the linear system
    local_energies = local_energies.flatten()
    de = local_energies - jnp.mean(local_energies)

    # At the moment the final vjp is centered by centering the auxiliary vector a.
    # This is the same as centering the jacobian but may have larger variance.
    dv = 2.0 * de / jnp.sqrt(N_mc)  # shape [N_mc,]
    if mode == "complex":
        dv = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)  # shape [N_mc,2]
    else:
        dv = jnp.real(dv)  # shape [N_mc,]

    if momentum is not None:
        if old_updates is None:
            old_updates = tree_map(jnp.zeros_like, parameters_real)
        else:
            acc = nkjax.apply_chunked(
                jvp_f_chunk, in_axes=(None, None, 0), chunk_size=chunk_size
            )(parameters_real, old_updates, samples)

            avg = jnp.mean(acc, axis=0)
            acc = (acc - avg) / jnp.sqrt(N_mc)
            dv -= momentum * acc

    if mode == "complex":
        dv = jax.lax.collapse(dv, 0, 2)  # shape [2*N_mc,] or [N_mc, ] if not complex

    # Generate random orthogonal projection vectors
    if projection_dim is None:
        projection_dim = n_params  # Use full rank if not specified
        projection_dim = N_mc

    # Use provided RNG key or create a new one
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Generate M random orthogonal vectors in parameter space
    # Q has shape (n_params, projection_dim)
    Q = _generate_random_orthogonal_vectors(rng_key, projection_dim, n_params)

    # Compute projected jacobian: for each of M vectors, compute JVP
    # This gives us a (N_mc, projection_dim) or (N_mc, 2, projection_dim) matrix
    def compute_projected_jacobian(parameters_real, projection_vectors, samples):
        """
        Compute the projected jacobian by doing JVPs along projection directions.

        Args:
            parameters_real: pytree of real parameters
            projection_vectors: (n_params, projection_dim) array
            samples: samples array

        Returns:
            Projected jacobian of shape (N_mc, projection_dim) or (N_mc, 2, projection_dim)
        """
        def jvp_single_direction(vec_flat):
            """Compute JVP along a single direction vector."""
            vec_pytree = unravel_fn(vec_flat)
            return jvp_f_chunk(parameters_real, vec_pytree, samples)

        # Apply to all projection directions
        # projection_vectors[:, i] is the i-th projection direction
        proj_jac = jax.vmap(jvp_single_direction, in_axes=1, out_axes=-1)(projection_vectors)
        return proj_jac  # shape: (N_mc, projection_dim) or (N_mc, 2, projection_dim)

    # Handle sharding for samples
    if config.netket_experimental_sharding:
        samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )

    # Compute the projected jacobian with chunking if specified
    if chunk_size is not None:
        # Chunk over samples
        samples_chunked, _ = nkjax.chunk(samples, chunk_size=chunk_size)
        proj_jac_chunks = jax.lax.map(
            lambda s_chunk: compute_projected_jacobian(parameters_real, Q, s_chunk),
            samples_chunked
        )
        # Concatenate chunks back together
        if mode == "complex":
            # proj_jac_chunks: (n_chunks, chunk_size, 2, projection_dim)
            proj_jac = jnp.concatenate(proj_jac_chunks, axis=0)  # (N_mc, 2, projection_dim)
        else:
            # proj_jac_chunks: (n_chunks, chunk_size, projection_dim)
            proj_jac = jnp.concatenate(proj_jac_chunks, axis=0)  # (N_mc, projection_dim)
    else:
        proj_jac = compute_projected_jacobian(parameters_real, Q, samples)

    # Center the projected jacobian
    # Shape: (N_mc, projection_dim) or (N_mc, 2, projection_dim)
    proj_jac_mean = jnp.mean(proj_jac, axis=0, keepdims=True)
    proj_jac_centered = (proj_jac - proj_jac_mean) / jnp.sqrt(N_mc)

    # Flatten for complex mode: (N_mc, 2, projection_dim) -> (2*N_mc, projection_dim)
    if mode == "complex":
        proj_jac_flat = jax.lax.collapse(proj_jac_centered, 0, 2)
    else:
        proj_jac_flat = proj_jac_centered

    # Compute NTK as O_proj @ O_proj.T
    # Shape: (N_mc, projection_dim) @ (projection_dim, N_mc) -> (N_mc, N_mc)
    # or (2*N_mc, projection_dim) @ (projection_dim, 2*N_mc) -> (2*N_mc, 2*N_mc)
    # Note: The NTK is already centered because we centered the projected jacobian
    ntk = proj_jac_flat @ proj_jac_flat.T

    # add diag shift
    ntk_shifted = ntk + diag_shift * jnp.eye(ntk.shape[0])

    # add projection regularization
    if proj_reg is not None:
        ntk_shifted = ntk_shifted + proj_reg / N_mc

    # some solvers return a tuple, some others do not.
    aus_vector = solver_fn(ntk_shifted, dv)
    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
    else:
        info = {}

    if info is None:
        info = {}

    # Compute the projected updates: updates_proj = O_proj.T @ aus_vector
    # Shape: (projection_dim, N_mc or 2*N_mc) @ (N_mc or 2*N_mc,) -> (projection_dim,)
    updates_proj = proj_jac_flat.T @ aus_vector

    # Project back to full parameter space: updates = Q @ updates_proj
    # Shape: (n_params, projection_dim) @ (projection_dim,) -> (n_params,)
    updates_flat = Q @ updates_proj

    # Unravel to pytree structure (in real representation)
    updates_real = unravel_fn(updates_flat)

    # Handle momentum in real representation
    if momentum is not None:
        updates_real = tree_map(lambda x, y: x + momentum * y, updates_real, old_updates)
        old_updates = updates_real

    # Convert back to original parameter structure (complex if needed)
    updates = rss(updates_real)

    return updates, old_updates, info
