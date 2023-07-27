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

from typing import Optional

import numpy as np
import numba

from netket.utils.types import Array, DType


class UnconstrainedHilbertIndex:
    def __init__(self, local_states, size):
        self._local_states = np.sort(local_states).astype(np.float64)
        self._local_size = len(self._local_states)
        self._size = size

        self._basis = np.zeros(size, dtype=np.int64)
        ba = 1
        for s in range(size):
            self._basis[s] = ba
            ba *= self._local_size

    def _local_state_number(self, x):
        return np.searchsorted(self.local_states, x)

    @property
    def size(self) -> int:
        return self._size

    @property
    def n_states(self):
        return self._local_size**self._size

    @property
    def local_states(self):
        return self._local_states

    @property
    def local_size(self) -> int:
        return self._local_size

    def states_to_numbers(
        self, states, out: Optional[Array] = None, dtype: DType = np.int32
    ):
        if states.ndim != 2:
            raise RuntimeError("Invalid input shape, expecting a 2d array.")

        if out is None:
            out = np.empty(states.shape[0], dtype)
        # else:
        #     assert out.size == states.shape[0]

        # Broadcasting to avoid inner loop
        state_numbers = self._local_state_number(states[:, ::-1])
        np.sum(state_numbers * self._basis, axis=-1, out=out)
        return out

    def numbers_to_states(
        self, numbers, out: Optional[Array] = None, dtype: DType = None
    ):
        if numbers.ndim != 1:
            raise RuntimeError("Invalid input shape, expecting a 1d array.")

        if out is None:
            out = np.empty((numbers.shape[0], self._size), dtype=dtype)
        # else:
        #     assert out.shape == (numbers.shape[0], self._size)

        _numbers_to_states(
            numbers, out, self._local_states, self._size, self._local_size
        )

        return out

    def all_states(self, out: Optional[Array] = None, dtype: DType = None):
        if out is None:
            out = np.full(
                (self.n_states, self._size), self._local_states[0], dtype=dtype
            )

        return self.numbers_to_states(np.arange(self.n_states, dtype=np.int32), out)


# TODO: Remove Numba and rewrite in numpy/jax
# This function is the last one that sill uses numba. Hopefully we
# can remove it one day


@numba.njit
def _numbers_to_states(numbers, out, _local_states, _size, _local_size):
    out.fill(_local_states[0])

    for i, number in enumerate(numbers):
        # _number_to_state(number, out[i], _local_states, _size, _local_size)
        out_i = out[i]
        ip = number
        k = _size - 1
        while ip > 0:
            out_i[k] = _local_states[ip % _local_size]
            ip = ip // _local_size
            k -= 1
