# Copyright 2022 The NetKet Authors - All rights reserved.
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

from numbers import Number
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import DType
from netket.jax import canonicalize_dtypes


class StaticRange(struct.Pytree):
    """
    An object representing a range similar to python's range, but that
    works with `jax.jit`.

    This range object can also be used to convert 'computational basis'
    configurations to integer indices ∈ [0,length].

    This object is used inside of Hilbert spaces.

    This object can be converted to a numpy or jax array:

    .. code-block:: python

        >>> import netket as nk; import numpy as np
        >>> n_max = 10
        >>> ran = nk.utils.StaticRange(start=0, step=1, length=n_max)
        >>> np.array(ran)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    And it can be used to convert between integer values starting at 0
    and the values in the range.

    .. code-block:: python

        >>> import netket as nk; import numpy as np
        >>> ran = nk.utils.StaticRange(start=-2, step=2, length=3)
        >>> np.array(ran)
        array([-2,  0,  2])
        >>> len(ran)
        3
        >>> ran.states_to_numbers(0)
        Array(1, dtype=int64)
        >>> ran.numbers_to_states(0)
        Array(-2, dtype=int64)

        >>> ran.numbers_to_states(1)
        Array(0, dtype=int64)

        >>> ran.numbers_to_states(2)
        Array(2, dtype=int64)

    """

    start: Number = struct.field(pytree_node=False)
    """The first value in the range."""
    step: Number = struct.field(pytree_node=False)
    """The difference between two consecutive values in the range."""
    length: int = struct.field(pytree_node=False)
    """The number of entries in the range."""
    dtype: DType = struct.field(pytree_node=False)
    """The dtype of the range."""

    def __init__(self, start: Number, step: Number, length: int, dtype: DType = None):
        """
        Constructs a Static Range object.

        To construct it, one must specify the start value, the step and the length.
        It is also possible to specify a `dtype`. In case it's not specified, it's
        inferred from the input arguments.

        For example, the :class:`~netket.utils.StaticRange` of a Fock Hilbert space
        is constructed as

        .. code-block:: python

            >>> import netket as nk
            >>> n_max = 10
            >>> nk.utils.StaticRange(start=0, step=1, length=n_max)
            StaticRange(start=0, step=1, length=10, dtype=int64)

        and the range of a Spin-1/2 Hilbert space is constructed as:

        .. code-block:: python

            >>> import netket as nk
            >>> n_max = 10
            >>> nk.utils.StaticRange(start=-1, step=2, length=2)
            StaticRange(start=-1, step=2, length=2, dtype=int64)

        Args:
            start: Value of the first entry
            step: Step between the entries
            length: Length of this range
            dtype: The data type
        """
        dtype = canonicalize_dtypes(start, step, dtype=dtype)

        self.start = np.array(start, dtype=dtype).item()
        self.step = np.array(step, dtype=dtype).item()
        self.length = int(length)

        self.dtype = dtype

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError
        return self.start + self.step * i

    def find(self, val):
        return int((val - self.start) / self.step)

    @partial(jax.jit, static_argnames="dtype")
    def states_to_numbers(self, x, dtype: DType = int):
        """Given an element in the range, returns it's index.

        Args:
            x: array of elements beloging to this range. No bounds checking
                is performed.
            dtype: Optional dtype to be used for the output.

        Returns:
            An array of integers, which can be.
        """
        idx = (x - self.start) / self.step
        if dtype is not None:
            idx = idx.astype(dtype)
        return idx

    @partial(jax.jit, static_argnames="dtype")
    def numbers_to_states(self, i, dtype: DType = None):
        """Given an integer index, returns the i-th elements in the range.

        Args:
            x: indices to extract from the range.
            dtype: Optional dtype to be used for the output.

        Returns:
            An array of values from the range. The dtype by default
            is that of the range.
        """

        if dtype is None:
            dtype = self.dtype
        start = jnp.array(self.start, dtype=dtype)
        step = jnp.array(self.step, dtype=dtype)
        return (start + step * i).astype(dtype)

    def flip_state(self, state):
        if not len(self) == 2:
            raise ValueError
        constant_sum = 2 * self.start + self.step
        return constant_sum - state

    def __array__(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        states = self.start + np.arange(self.length, dtype=dtype) * self.step
        return states.astype(dtype)

    def __hash__(self):
        return hash(("StaticRange", self.start, self.step, self.length))

    def __eq__(self, o):
        if isinstance(o, StaticRange):
            return (
                self.start == o.start
                and self.step == o.step
                and self.length == o.length
            )
        else:
            return self.__array__() == o

    def __repr__(self):
        return f"StaticRange(start={self.start}, step={self.step}, length={self.length}, dtype={self.dtype})"
