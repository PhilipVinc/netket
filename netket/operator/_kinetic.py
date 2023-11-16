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
from typing import Optional, Union
from collections.abc import Iterable

import numpy as np

import jax
import jax.numpy as jnp

from netket.utils.types import DType
from netket.utils.numbers import is_scalar
from netket.hilbert import AbstractHilbert

from ._laplacian import Laplacian


class KineticEnergy(Laplacian):
    r"""This is the kinetic energy operator (hbar = 1). The local value is given by:
    :math:`E_{kin} = -1/2 ( \sum_i \frac{1}{m_i} (\log(\psi))'^2 + (\log(\psi))'' )`
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        sites: Union[int, Iterable[int]] = None,
        *,
        mass: Union[float, Iterable[float]] = 1,
        dtype: Optional[DType] = None,
        algorithm: str = "bwd-fwd",
    ):
        r"""
        Constructs the Kinetic term on a given site.

        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            mass: float if all masses are the same, list indicating the mass of each particle otherwise
            dtype: Data type of the matrix elements. Defaults to `np.float64`
            use_jet: (Experimental!) Compute the derivatives using Taylor Polynomials.
        """
        if sites is None:
            sites = range(hilbert.size)

        if not isinstance(sites, Iterable):
            sites = [sites]

        if is_scalar(mass):
            mass = np.full((len(sites),), mass, dtype=dtype)
        else:
            mass = np.asarray(mass, dtype=dtype)

        if not all(s in range(hilbert.size) for s in sites):
            raise ValueError("all sites should be between 0 and hilbert.size -1")

        if len(set(sites)) != len(sites):
            raise ValueError(
                "There are repeated sites. There should not be repeated sites."
            )

        if len(mass) != len(sites):
            raise ValueError(
                "mass should be a scalar or have the same length of sites."
            )

        coeffs = jnp.where(mass != 0, jnp.reciprocal(mass), 0.0)
        super().__init__(
            hilbert, sites, coeffs=coeffs, dtype=dtype, algorithm=algorithm
        )

    @property
    def mass(self) -> jax.Array:
        return jnp.where(self.coeffs != 0, jnp.reciprocal(self.coeffs), 0.0)

    def __repr__(self):
        return f"KineticEnergy(acting_on={self.acting_on}, masses={self.mass})"
