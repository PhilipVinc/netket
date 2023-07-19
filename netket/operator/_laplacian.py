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

from typing import Callable, Hashable, Iterable, Optional, Tuple, Union
from numbers import Number

import numpy as np

import jax
import jax.numpy as jnp

from netket import jax as nkjax

from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import struct, HashableArray
from netket.utils.types import DType, PyTree, Array
from netket.utils.numbers import is_scalar


def jacrev(f):
    def jacfun(x):
        y, vjp_fun = nkjax.vjp(f, x)
        if y.size == 1:
            eye = jnp.eye(y.size, dtype=x.dtype)[0]
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
        else:
            eye = jnp.eye(y.size, dtype=x.dtype)
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
        return J

    return jacfun


def jacfwd(f):
    def jacfun(x):
        jvp_fun = lambda s: jax.jvp(f, (x,), (s,))[1]
        eye = jnp.eye(len(x), dtype=x.dtype)
        J = jax.vmap(jvp_fun, in_axes=0)(eye)
        return J

    return jacfun


@struct.dataclass
class LaplacianOperatorPyTree:
    """Internal class used to pass data from the operator to the jax kernel.

    This is used such that we can pass a PyTree containing some static data.
    We could avoid this if the operator itself was a pytree, but as this is not
    the case we need to pass as a separte object all fields that are used in
    the kernel.

    We could forego this, but then the kernel could not be marked as
    @staticmethod and we would recompile every time we construct a new operator,
    even if it is identical
    """

    hilbert: AbstractHilbert = struct.field(pytree_node=False)
    algorithm: Callable = struct.field(pytree_node=False)
    acting_on: Array
    coeffs: Array


class Laplacian(ContinuousOperator):
    r"""This is the laplacian operator on a set of coordinates :math:`i`,
    :math:`\sum_i\alpha_i\nabla_{x_i}^2` and coefficient :math:`\alpha_i`.

    For N particles in an D-dimensional space, the coefficients :math:`i`
    label one of the :math:`ND` degrees of freedom, which are usually
    ordered as :math:`x^{(1)},y^{(1)},z^{(1)}, x^{2}, y^{(2)}...`, but you
    should check the Hilbert space definition for more details.

    The local estimator is computed as:

    .. math:

        Laplacian^text{loc} = \frac{\sum_i\alpha_i \nabla_{x_i}^2\psi(x)}{\psi(x)}
            = \sum_i \alpha_i (\partial_x\log\psi(x))^2 + (\partial_x^2\log\psi(x))


    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        dof: Union[int, Iterable[int]],
        *,
        coeffs: Union[Number, Iterable[Number]] = 1,
        dtype: Optional[DType] = None,
        algorithm: str = "bwd-fwd",
    ):
        r"""
        Constructs the Laplacian acting on a set of degrees of freedom with a certain
        coefficient.

        The experimental attribute `algorithm` can be used to specify one of
        several possible implementations of the Laplacian. Please notice that this
        attribute is experimental and might be removed without warning in future
        releases, and specifying it might lead to bugs or wrong results.
        The possible implementations are:

         - `algorithm = "bwd-fwd"` (default). This computes the laplacian using
           forward-over-backward mode automatic-differentiation.
         - `algorithm = "fwd"` or `"fwd-fwd"`; This computes the laplacian
           using forward-over-forward mode automatic-differentiation.
         - `algorithm = "jet"` or `"taylor"`; This computes the jacobian and
           laplacian simultaneously using Taylor-mode higher-order automatic
           differentiation. Note that this mode is highly experimental and might not
           support all functions.

        Args:
            hilbert: The underlying Hilbert space on which the operator is defined.
            dof: integer or iterable of integers listing the degrees of freedom of
                the hilbert space upon which this operator acts. Every degree of
                freedom must be an integer in :math:`0\leq\text{dof}<\text{hilbert.size}`.
            coeff: coefficient or list of coefficients in the linear combination of
                the laplacians of every degree of freedom. If it is a single number,
                it is assumed to be identical for all degrees of freedom. If it is an
                iterable, it must have the same length as `dof`.
            dtype: Data type of the matrix elements. If unspecified it is inferred from
                the dtype of `coeff`.
            algorithm: (Experimental!) Specify the particular AD algorithm used
                to compute the laplacian.
        """
        if not isinstance(dof, Iterable):
            dof = [dof]

        if is_scalar(coeffs):
            coeffs = np.full((len(dof),), coeffs, dtype=dtype)
        else:
            coeffs = np.asarray(coeffs, dtype=dtype)

        if not all(s in range(hilbert.size) for s in dof):
            raise ValueError("all dof should be between 0 and hilbert.size -1")

        if len(set(dof)) != len(dof):
            raise ValueError(
                "There are repeated dof. There should not be repeated dof."
            )

        if len(coeffs) != len(dof):
            raise ValueError(
                "coeffs should be a scalar or have the same length of dof."
            )

        if algorithm not in ["fwd", "fwd-fwd", "bwd", "bwd-fwd", "jet", "taylor"]:
            raise TypeError(
                "`algorithm` must be one of 'fwd-fwd', 'bwd-fwd' or 'taylor'."
            )

        super().__init__(hilbert, coeffs.dtype)

        self._state_dict = {s: m for s, m in zip(dof, coeffs)}
        self._algorithm = algorithm
        self._initialized = False
        self.__attrs = None

    def _setup(self):
        # self._mask = tuple(i in sites for i in range(self.hilbert.size))

        coeffs = np.zeros((self.hilbert.size,), self.dtype)
        for s, m in self._state_dict.items():
            coeffs[s] = m

        self._coeffs = coeffs
        self._coeffs_jax = jnp.array(self.coeffs, dtype=self.dtype)

        self._is_hermitian = np.allclose(coeffs.imag, 0.0)
        self._initialized = True

    def __add__(self, other):
        if isinstance(other, Laplacian):
            if not self.hilbert == other.hilbert:
                raise ValueError("Cannot add Laplacians on different Hilbert spaces")

            new_state = self._state_dict.copy()
            for k, v in other._state_dict.items():
                new_state[k] = new_state.get(k, 0) + v
            sites = list(new_state.keys())
            coeffs = list(new_state.values())
            return Laplacian(self.hilbert, sites, coeffs=coeffs)

        return super().__add__(other)

    @property
    def acting_on(self):
        return tuple(self._state_dict.keys())

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def coeffs(self):
        return np.asarray(list(self._state_dict.values()), dtype=self.dtype)

    @property
    def is_hermitian(self):
        if not self._initialized:
            self._setup()
        return self._is_hermitian

    @staticmethod
    def _expect_kernel(
        logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        vmap_kernel = jax.vmap(
            _laplacian_expect_kernel_single, in_axes=(None, None, 0, None)
        )
        return vmap_kernel(logpsi, params, x, data)

    def _pack_arguments(self) -> PyTree:
        if not self._initialized:
            self._setup()

        return LaplacianOperatorPyTree(
            self.hilbert, self.algorithm, self.acting_on, self._coeffs_jax
        )

    @property
    def _attrs(self) -> Tuple[Hashable, ...]:
        print("_attrs of laplacan", self)
        if self.__attrs is None:
            self.__attrs = (
                self.hilbert,
                self.acting_on,
                HashableArray(self.coeffs),
                self.dtype,
            )
        return self.__attrs

    def __repr__(self):
        return f"Laplacian(acting_on={self.acting_on}, coeffs={self.coeffs})"

    def __truediv__(self, other):
        if not is_scalar(other):
            raise TypeError("Only division by a scalar number is supported.")

        if other == 0:
            raise ValueError("Dividing by 0")
        return self.__mul__(1.0 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if is_scalar(other):
            # op = self.copy(dtype=np.promote_types(self.dtype, _dtype(other)))
            # return op.__imul__(other)
            new_state = {k: other * v for k, v in self._state_dict.items()}
            return Laplacian(
                self.hilbert, list(new_state.keys()), coeffs=list(new_state.values())
            )
        return NotImplemented

    def __neg__(self):
        new_state = {k: -v for k, v in self._state_dict.items()}
        return Laplacian(
            self.hilbert, list(new_state.keys()), coeffs=list(new_state.values())
        )

    def __sub__(self, other):
        if isinstance(other, Laplacian):
            return self + (-other)
        return super().__sub__(other)


def _laplacian_expect_kernel_single(
    logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
):
    def logpsi_x(x):
        return logpsi(params, x)

    if data.algorithm.lower() in ["bwd", "bwd-fwd"]:
        dlogpsi_x = jacrev(logpsi_x)
        dp_dx2 = jnp.diag(jacfwd(dlogpsi_x)(x)[0].reshape(x.shape[0], x.shape[0]))
        dp_dx = dlogpsi_x(x)[0][0] ** 2
    elif data.algorithm.lower() in ["fwd", "fwd-fwd"]:
        dlogpsi_x = jacfwd(logpsi_x)
        dp_dx2 = jnp.diag(jacfwd(dlogpsi_x)(x)[0].reshape(x.shape[0], x.shape[0]))
        dp_dx = dlogpsi_x(x)[0][0] ** 2
    elif data.algorithm.lower() in ["jet", "taylor"]:
        from jax.experimental.jet import jet

        xI = jnp.eye(x.size)

        # jet returns the taylor coefficients at different orders
        _, (f1, f2) = jax.vmap(
            lambda h0, h1: jet(logpsi_x, (x,), ((h0, h1),)),
            in_axes=(0, 0),
            out_axes=(None, [0, 0]),
        )(xI, xI)
        dp_dx = f1**2
        dp_dx2 = f2 - f1

    kin_term = dp_dx2 + dp_dx

    # only cut a part of the hilbert space if we really need it...
    if len(data.acting_on) != data.hilbert.size:
        kin_term = kin_term[..., list(data.acting_on)]

    return -0.5 * jnp.sum(data.coeffs * kin_term, axis=-1)
