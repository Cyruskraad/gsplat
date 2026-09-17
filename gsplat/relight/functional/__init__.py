# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stateless building blocks of the ATLAS relighting model.

Every function here is pure, shape-checked, device- and dtype-agnostic, and runs
on CPU. None of them touch the CUDA backend, which is what lets the whole
correctness argument be tested without a GPU.
"""

from .atoms import (
    atom_gram_matrix,
    default_sharpness,
    equirect_directions,
    equirect_solid_angles,
    evaluate_atoms,
    fibonacci_sphere,
    make_sg_atoms,
    project_environment,
    project_point_light,
)
from .compress import (
    CompressedTransport,
    TransportSpectrum,
    compress_transport,
    energy_retained,
    project_light_to_compressed,
    rank_for_energy,
    transport_spectrum,
)
from .inverse import LightSolution, normal_equations, solve_light
from .nearfield import (
    cosine_power_profile,
    incident_radiance,
    inverse_square_falloff,
    light_directions,
)
from .prefilter import (
    combine_prefiltered,
    prefilter_atoms,
    prefilter_equirect,
    roughness_to_sharpness,
)
from .transport import (
    composite,
    compositing_weights,
    contract,
    contract_screen,
    pack_transport,
    unpack_transport,
)

__all__ = [
    # atoms
    "atom_gram_matrix",
    "default_sharpness",
    "equirect_directions",
    "equirect_solid_angles",
    "evaluate_atoms",
    "fibonacci_sphere",
    "make_sg_atoms",
    "project_environment",
    "project_point_light",
    # transport
    "composite",
    "compositing_weights",
    "contract",
    "contract_screen",
    "pack_transport",
    "unpack_transport",
    # near field
    "cosine_power_profile",
    "incident_radiance",
    "inverse_square_falloff",
    "light_directions",
    # prefilter
    "combine_prefiltered",
    "prefilter_atoms",
    "prefilter_equirect",
    "roughness_to_sharpness",
    # compress
    "CompressedTransport",
    "TransportSpectrum",
    "compress_transport",
    "energy_retained",
    "project_light_to_compressed",
    "rank_for_energy",
    "transport_spectrum",
    # inverse lighting
    "LightSolution",
    "normal_equations",
    "solve_light",
]
