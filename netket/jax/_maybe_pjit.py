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

from typing import Callable, Optional, Any, Sequence, Iterable, Tuple, Union

import jax
from jax.experimental import pjit
from jax.experimental import maps
from jax.experimental import mesh_utils

from jax.sharding import PositionalSharding

from netket import config

def pmap(fun: Callable,
    *,
    static_argnums: Union[int, Iterable[int]] = None,
    static_argnames = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    ):

    return jax.jit(fun, static_argnums=static_argnums, static_argnames=static_argnames)