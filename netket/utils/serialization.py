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

from warnings import warn

import jax
import numpy as np

from flax import serialization

from netket.utils import struct, mpi
from netket.utils.types import Array, PyTree


class MPISerializedArray(jax.xla.DeviceArray):
    def __init__(self, arr):
        self.arr = arr

    def __repr__(self):
        return f"MPIArray({repr(self.arr)})"

    def __str__(self):
        return f"MPIArray({str(self.arr)})"

    def __array__(self, dtype=None):
        if mpi.n_nodes > 1:
            if isinstance(self.arr, jax.xla.DeviceArray):
                self.arr.block_until_ready()

            return mpi.gather(np.asarray(self.arr, dtype=dtype), 0)
        else:
            return np.asarray(self.arr, dtype=dtype)


@struct.dataclass
class MPIShard:
    val: Array

    @property
    def shape(self):
        return self.val.shape

    @property
    def ndim(self):
        return self.val.ndim

    def __repr__(self):
        return (
            f"MPIShard(shape={self.val.shape}, dtype={self.val.dtype}, "
            f"val={self.val})"
        )


def MPIShardLeaves(tree: PyTree) -> PyTree:
    """
    Use to signal during serialisation that the child object of this one is
    sharded among different MPI Nodes.

    Behaves **very** similarly to a `jax.GlobalShardedDeviceArray`.
    """
    return jax.tree_map(lambda x: MPIShard(x), tree)


def MPIUnshardLeaves(target, tree: PyTree) -> PyTree:
    return jax.tree_map(lambda x, y: y.val, target, tree)


def serialize_MPIShard(shard: MPIShard):
    print(f"Serializing {shard}")
    return {
        "arr": MPISerializedArray(shard.val),
        "ndim": serialization.to_state_dict(shard.ndim),
        "n_nodes": serialization.to_state_dict(mpi.n_nodes),
    }


def deserialize_MPIShard(shard, state_dict):
    print(f"deserialize_MPIShard {state_dict}")
    ndim = serialization.from_state_dict(shard.ndim, state_dict["ndim"])
    arr = state_dict["arr"]
    if arr.ndim == ndim:
        if state_dict["n_nodes"] == 1:
            return MPIShard(arr)
        else:
            raise NotImplementedError()
    else:
        if mpi.rank < state_dict["n_nodes"]:
            return MPIShard(arr[mpi.rank])
        else:
            warn(f"rank {mpi.rank} failed to reco")
            return MPIShard(arr[0])


serialization.register_serialization_state(
    MPIShard, serialize_MPIShard, deserialize_MPIShard, override=True
)
