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

import jax
import numpy as np


class Range:
    """
    An object representing a range similar to python's range, but that
    works with `jax.jit` and can be used within Numba-blocks.

    This range object can also be used to convert 'computational basis'
    configurations to integer indices ∈ [0,length].
    """

    def __init__(self, start, step, length):
        """
        Constructs a Static Range object.

        Args:
            start: Value of the first entry
            step: step between the entries
            length: Length of this range
        """
        self.start = start
        self.step = step
        self.length = length
        self.end = start + step * length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError
        return self.start + self.step * i

    def find(self, val):
        return int((val - self.start) / self.step)

    def basis_to_index(self, x):
        return ((x - self.start) / self.step).astype(np.int32)

    def index_to_basis(self, i):
        return self.start + self.step * i

    def flip_state(self, state):
        if not len(self) == 2:
            raise ValueError
        constant_sum = 2 * self.start + self.step
        return constant_sum - state

    def __array__(self, dtype=None):
        return self.start + np.arange(self.length, dtype=dtype) * self.step

    def __hash__(self):
        return hash(("StaticRange", self.start, self.step, self.length))

    def __eq__(self, o):
        if isinstance(o, Range):
            return self.start == o.start and self.step == o.step and self.end == o.end
        return False

    def __repr__(self):
        return (
            f"StaticRange(start={self.start}, step={self.step}, length={self.length})"
        )


# support for jax pytree flattening unflattening
def iterate_range(x):
    meta = (
        x.start,
        x.step,
        x.length,
    )
    data = ()
    return data, meta


def range_from_iterable(meta, data):
    return Range(*meta)


jax.tree_util.register_pytree_node(Range, iterate_range, range_from_iterable)
