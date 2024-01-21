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

from netket.sampler.rules.base import MetropolisRule

from netket.sampler.rules.fixed import FixedRule
from netket.sampler.rules.local import LocalRule
from netket.sampler.rules.exchange import ExchangeRule
from netket.sampler.rules.hamiltonian import HamiltonianRule
from netket.sampler.rules.continuous_gaussian import GaussianRule
from netket.sampler.rules.langevin import LangevinRule
from netket.sampler.rules.tensor import TensorRule
from netket.sampler.rules.multiple import MultipleRules

# numpy backend
from netket.sampler.rules.local_numpy import LocalRuleNumpy
from netket.sampler.rules.hamiltonian_numpy import HamiltonianRuleNumpy
from netket.sampler.rules.custom_numpy import CustomRuleNumpy

from netket.utils import _hide_submodules

_hide_submodules(__name__)
