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

"""Near-field point lights, and the gap they open.

Training observations come from a flash roughly a metre away. Deployment is
under a distant environment map. The two differ in three ways that are all
analytic, so all three are divided out here rather than left for the network to
absorb:

1. **Direction.** Every primitive sees its own direction to a near light. Under
   a distant environment they all see the same directions. So the atoms are
   defined over *direction* and the per-primitive direction is computed here.
2. **Falloff.** Irradiance from a point source falls as ``1/d^2``. A distant
   environment has no falloff.
3. **Emitter profile.** A real flash is not isotropic. Its angular profile is
   measured once against a white reference card and applied here.

What remains after dividing all three out is a transport function of direction
alone, which is the thing the atom basis represents and the thing that
transfers to an environment map.

This is the method's principal scientific risk, and it is deliberately
concentrated in this one module so that it can be tested in isolation and so
that turning the model off (``--no-nearfield``) is a one-line ablation. The
end-to-end check is not in here and cannot be: it is an HDR environment probe
plus a photograph of the object under that same environment, compared against
the relit render.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "light_directions",
    "inverse_square_falloff",
    "cosine_power_profile",
    "incident_radiance",
]


def light_directions(
    means: Tensor,
    light_position: Tensor,
    *,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Unit directions from primitives towards a point light, and distances.

    Args:
        means: ``[N, 3]`` primitive centres in world space.
        light_position: ``[3]`` light position in world space.
        eps: Floor on the distance, guarding a primitive that coincides with
            the light. A coincident primitive is a degenerate capture, not a
            numerical detail, so the floor is small and deliberate.

    Returns:
        ``(directions, distances)`` with shapes ``[N, 3]`` and ``[N]``.
    """
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"means must be [N, 3], got {tuple(means.shape)}")
    if light_position.shape != (3,):
        raise ValueError(
            f"light_position must be [3], got {tuple(light_position.shape)}"
        )
    offset = light_position.to(means.dtype).to(means.device) - means  # [N, 3]
    distances = torch.linalg.norm(offset, dim=-1).clamp(min=eps)  # [N]
    return offset / distances.unsqueeze(-1), distances


def inverse_square_falloff(distances: Tensor, *, reference_distance: float = 1.0):
    """``(reference_distance / d)^2``.

    Normalised at ``reference_distance`` so that intensity is expressed in
    radiance at a stated distance rather than in raw candela, which keeps the
    fitted intensity on the same scale as the image data.

    Args:
        distances: ``[...]`` positive distances.
        reference_distance: Where the falloff equals 1.

    Returns:
        ``[...]`` falloff factors.
    """
    if reference_distance <= 0.0:
        raise ValueError(f"reference_distance must be > 0, got {reference_distance}")
    if bool((distances <= 0).any()):
        raise ValueError("distances must be positive")
    return (reference_distance / distances) ** 2


def cosine_power_profile(
    directions: Tensor,
    light_axis: Tensor,
    exponent: float,
) -> Tensor:
    """Angular emission profile of the flash, ``max(0, -d . axis)^n``.

    ``directions`` point *from the surface towards the light*, so the direction
    of travel of the light is their negation; ``light_axis`` is where the flash
    is aimed. An exponent of 0 is an isotropic emitter.

    Args:
        directions: ``[N, 3]`` unit vectors towards the light.
        light_axis: ``[3]`` unit vector along the flash's optical axis.
        exponent: Non-negative cosine power.

    Returns:
        ``[N]`` profile in ``[0, 1]``.
    """
    if exponent < 0.0:
        raise ValueError(f"exponent must be >= 0, got {exponent}")
    if light_axis.shape != (3,):
        raise ValueError(f"light_axis must be [3], got {tuple(light_axis.shape)}")
    if exponent == 0.0:
        return torch.ones(
            directions.shape[:-1], device=directions.device, dtype=directions.dtype
        )
    axis = light_axis.to(directions.dtype).to(directions.device)
    cos = (-directions * axis).sum(dim=-1).clamp(min=0.0)
    return cos**exponent


def incident_radiance(
    means: Tensor,
    light_position: Tensor,
    intensity: Tensor,
    *,
    light_axis: Optional[Tensor] = None,
    profile_exponent: float = 0.0,
    reference_distance: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Per-primitive direction to a near-field flash and the radiance it delivers.

    Feed the result to
    :func:`atlas.functional.atoms.project_point_light` to obtain the
    per-primitive light coefficients used in training.

    Args:
        means: ``[N, 3]`` primitive centres.
        light_position: ``[3]``.
        intensity: ``[3]`` RGB radiant intensity at ``reference_distance`` on
            the flash axis.
        light_axis: ``[3]`` flash aim. Required when ``profile_exponent > 0``.
        profile_exponent: Cosine power of the emission profile.
        reference_distance: Distance at which ``intensity`` is stated.

    Returns:
        ``(directions [N, 3], radiance [N, 3])``.
    """
    if intensity.shape != (3,):
        raise ValueError(f"intensity must be [3], got {tuple(intensity.shape)}")
    directions, distances = light_directions(means, light_position)
    falloff = inverse_square_falloff(
        distances, reference_distance=reference_distance
    )  # [N]
    if profile_exponent > 0.0:
        if light_axis is None:
            raise ValueError("light_axis is required when profile_exponent > 0")
        falloff = falloff * cosine_power_profile(
            directions, light_axis, profile_exponent
        )
    radiance = intensity.to(means.dtype).to(means.device) * falloff.unsqueeze(-1)
    return directions, radiance
