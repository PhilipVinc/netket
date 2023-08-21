# Copyright 2023 The NetKet Authors - All rights reserved.
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

import numpy as np

from . import config


def default_dtype_floating():
    if config.netket_default_dtype_bits == 32:
        return np.float32
    else:
        return np.float64


def default_dtype_complex():
    if config.netket_default_dtype_bits == 32:
        return np.complex64
    else:
        return np.complex128


def default_dtype_integer():
    if config.netket_default_dtype_bits == 32:
        return np.int32
    else:
        return np.int64
