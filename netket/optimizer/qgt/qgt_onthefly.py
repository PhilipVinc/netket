# Copyright 2021 The NetKet Authors - All rights reserved.
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

from typing import Callable, Optional, Union
from functools import partial

import jax
from jax import numpy as jnp
from flax import struct

import netket.jax as nkjax
from netket.utils.types import Array, PyTree

from netket.nn import split_array_mpi

from .common import check_valid_vector_type
from .qgt_onthefly_logic import mat_vec_factory, mat_vec_chunked_factory

from ..linear_operator import LinearOperator, Uninitialized


def QGTOnTheFly(vstate=None, *, chunk_size=None, **kwargs) -> "QGTOnTheFlyT":
    """
    Lazy representation of an S Matrix computed by performing 2 jvp
    and 1 vjp products, using the variational state's model, the
    samples that have already been computed, and the vector.

    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
    the field `sr`.

    Args:
        vstate: The variational State.
        chunk_size: If supplied, overrides the chunk size of the variational state
                    (useful for models where the backward pass requires more
                    memory than the forward pass).
    """
    if vstate is None:
        return partial(QGTOnTheFly, chunk_size=chunk_size, **kwargs)

    if kwargs.pop("diag_scale", None) is not None:
        raise NotImplementedError(
            "\n`diag_scale` argument is not yet supported by QGTOnTheFly."
            "Please use `QGTJacobianPyTree` or `QGTJacobianDense`.\n\n"
            "You are also encouraged to nag the developers to support "
            "this feature.\n\n"
        )

    # TODO: Find a better way to handle this case
    from netket.vqs import ExactState

    if isinstance(vstate, ExactState):
        samples = split_array_mpi(vstate._all_states)
        pdf = split_array_mpi(vstate.probability_distribution())
    else:
        if jnp.ndim(vstate.samples) == 2:
            samples = vstate.samples
        else:
            samples = vstate.samples.reshape((-1, vstate.samples.shape[-1]))
        pdf = None

    if chunk_size is None and hasattr(vstate, "chunk_size"):
        chunk_size = vstate.chunk_size

    n_samples = samples.shape[0]

    if chunk_size is None or chunk_size >= n_samples:
        mv_factory = mat_vec_factory
        chunking = False
    else:
        samples, _ = nkjax.chunk(samples, chunk_size)
        if pdf is not None:
            pdf, _ = nkjax.chunk(pdf, chunk_size)
        mv_factory = mat_vec_chunked_factory
        chunking = True

    mat_vec = mv_factory(
        forward_fn=vstate._apply_fun,
        params=vstate.parameters,
        model_state=vstate.model_state,
        samples=samples,
        pdf=pdf,
    )
    return QGTOnTheFlyT(
        _mat_vec=mat_vec,
        _params=vstate.parameters,
        _chunking=chunking,
        **kwargs,
    )


@struct.dataclass
class QGTOnTheFlyT(LinearOperator):
    """
    Lazy representation of an S Matrix computed by performing 2 jvp
    and 1 vjp products, using the variational state's model, the
    samples that have already been computed, and the vector.

    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
    the field `sr`.
    """

    _mat_vec: Callable[[PyTree, float], PyTree] = Uninitialized
    """The S matrix-vector product as generated by mat_vec_factory.
    It's a jax.Partial, so can be used as pytree_node."""

    _params: PyTree = Uninitialized
    """The first input to apply_fun (parameters of the ansatz).
    Only used as a shape placeholder."""

    _chunking: bool = struct.field(pytree_node=False, default=False)
    """Whether the implementation with chunks is used which currently does not support vmapping over it"""

    @jax.jit
    def __matmul__(self, vec: Union[PyTree, Array]) -> Union[PyTree, Array]:
        """
        Perform the lazy mat-vec product, where vec is either a tree with the same structure as
        params or a ravelled vector
        """
        # if has a ndim it's an array and not a pytree
        if hasattr(vec, "ndim"):
            if not vec.ndim == 1:
                raise ValueError("Unsupported mat-vec for chunks of vectors")
            # If the input is a vector
            if not nkjax.tree_size(self._params) == vec.size:
                raise ValueError(
                    f"""Size mismatch between number of parameters ({nkjax.tree_size(self.params)})
                        and vector size {vec.size}.
                     """
                )

            _, unravel = nkjax.tree_ravel(self._params)
            vec = unravel(vec)
            ravel_result = True
        else:
            ravel_result = False

        check_valid_vector_type(self._params, vec)

        vec = nkjax.tree_cast(vec, self._params)

        res = self._mat_vec(vec, self.diag_shift)

        if ravel_result:
            res, _ = nkjax.tree_ravel(res)

        return res

    @jax.jit
    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree], **kwargs) -> PyTree:
        check_valid_vector_type(self._params, y)

        y = nkjax.tree_cast(y, self._params)

        # we could cache this...
        if x0 is None:
            x0 = jax.tree_map(jnp.zeros_like, y)

        out, info = solve_fun(self, y, x0=x0)
        return out, info

    @jax.jit
    def _to_dense(self) -> Array:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        Npars = nkjax.tree_size(self._params)
        I = jax.numpy.eye(Npars)

        if self._chunking:
            # the linear_call in mat_vec_chunked does currently not have a jax batching rule,
            # so it cannot be vmapped but we can use scan
            # which is better for reducing the memory consumption anyway
            _, out = jax.lax.scan(lambda _, x: (None, self @ x), None, I)
        else:
            out = jax.vmap(lambda x: self @ x, in_axes=0)(I)

        if jnp.iscomplexobj(out):
            out = out.T

        return out

    def __repr__(self):
        return f"QGTOnTheFly(diag_shift={self.diag_shift})"
