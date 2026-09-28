# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
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

"""A genuine microfacet BRDF, used only to generate ground truth to test against.

Nothing in the trained pipeline evaluates this at runtime -- the whole point of
:mod:`atlas.functional.specular` is that a *prefiltered* lookup replaces it. This
module exists so the synthetic generator can produce a surface response that is
truly a function of both the incident **and** outgoing direction, independently
of the spherical-Gaussian atoms the model is fitted in. Measuring the trained
model against a target built from its own basis would only ever measure the fit;
measuring it against Cook-Torrance/GGX measures the thing that matters, which is
whether the representation captures gloss at all.

Standard Trowbridge-Reitz (GGX) with height-correlated Smith masking-shadowing
and Schlick's Fresnel approximation -- the combination real-time renderers
settled on, e.g. Karis, *Real Shading in Unreal Engine 4* (2013). The roughness
parameterisation is Disney's ``alpha = roughness^2``, the same convention
:func:`atlas.functional.prefilter.roughness_to_sharpness` already uses, so a
roughness value means the same thing on both sides of a comparison between them.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "alpha_from_roughness",
    "half_vector",
    "ggx_distribution",
    "smith_ggx_visibility",
    "fresnel_schlick",
    "ggx_specular",
    "microfacet_response",
]


def alpha_from_roughness(roughness: Tensor | float) -> Tensor:
    """Disney's ``alpha = roughness^2``, as a tensor for autograd to flow through."""
    value = roughness if torch.is_tensor(roughness) else torch.tensor(roughness)
    return value * value


def half_vector(view_directions: Tensor, light_directions: Tensor) -> Tensor:
    """Unit ``normalize(w_i + w_o)``, the axis the NDF is evaluated about.

    Both arguments point *away* from the surface, the same convention
    :func:`atlas.functional.calibration.reflect` and
    :func:`atlas.functional.specular.reflect` use. When the two coincide -- the
    co-located capture case -- this is simply the shared direction itself, which
    is the algebraic reason a flash-on-camera orbit samples ``D(n . w)`` and
    nothing else: see the module docstring of
    :mod:`atlas.functional.specular`.
    """
    if view_directions.shape != light_directions.shape:
        raise ValueError(
            f"view_directions {tuple(view_directions.shape)} and "
            f"light_directions {tuple(light_directions.shape)} must match"
        )
    summed = view_directions + light_directions
    return summed / summed.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def ggx_distribution(cos_nh: Tensor, alpha: Tensor) -> Tensor:
    """The Trowbridge-Reitz normal distribution function, ``D(h)``.

    .. math::
        D = \\frac{\\alpha^2}{\\pi \\big((n \\cdot h)^2 (\\alpha^2 - 1) + 1\\big)^2}

    Peak-normalised so that ``D(0) `` (the mirror direction) integrates to a
    finite, roughness-dependent value rather than a delta -- the reason a
    prefiltered lookup can approximate it at all, and the reason
    :func:`atlas.functional.specular.min_resolvable_roughness` has to exist: at
    small ``alpha`` this concentrates in a solid angle no fixed-resolution map
    can resolve.

    Args:
        cos_nh: ``[...]`` cosine between the normal and the half-vector, any
            sign (clamped to zero below the horizon).
        alpha: ``[...]`` or scalar, from :func:`alpha_from_roughness`.

    Returns:
        ``[...]``, non-negative.
    """
    cos_nh = cos_nh.clamp_min(0.0)
    alpha2 = (alpha * alpha).clamp_min(1e-10)
    denominator = cos_nh * cos_nh * (alpha2 - 1.0) + 1.0
    return alpha2 / (math.pi * (denominator * denominator).clamp_min(1e-12))


def smith_ggx_visibility(cos_nv: Tensor, cos_nl: Tensor, alpha: Tensor) -> Tensor:
    """Height-correlated Smith masking-shadowing, folded with the BRDF's
    ``1 / (4 cos_nv cos_nl)`` normalisation into a single **visibility** term.

    Returns ``G / (4 cos_nv cos_nl)`` directly (Karis 2013, "Vis" rather than
    "G"), because that quotient is finite and well-conditioned even as either
    cosine approaches zero, while the two factors computed separately are not.

    Args:
        cos_nv: ``[...]`` cosine between the normal and the view direction.
        cos_nl: ``[...]`` cosine between the normal and the light direction.
        alpha: ``[...]`` or scalar.

    Returns:
        ``[...]``, non-negative.
    """
    cos_nv = cos_nv.clamp_min(0.0)
    cos_nl = cos_nl.clamp_min(0.0)
    k = alpha * 0.5  # height-correlated form; Karis eq. for direct lighting
    ggx_v = cos_nv * (1.0 - k) + k
    ggx_l = cos_nl * (1.0 - k) + k
    denominator = (4.0 * ggx_v * ggx_l).clamp_min(1e-10)
    return (cos_nv * cos_nl) / denominator


def fresnel_schlick(cos_vh: Tensor, f0: Tensor) -> Tensor:
    """Schlick's approximation, ``F0 + (1 - F0)(1 - cos)^5``.

    Args:
        cos_vh: ``[...]`` cosine between the view direction and the half-vector
            (equal to the light/half-vector cosine by construction).
        f0: ``[...]`` or scalar reflectance at normal incidence, in ``[0, 1]``.

    Returns:
        ``[...]``, in ``[f0, 1]``.
    """
    cos_vh = cos_vh.clamp(0.0, 1.0)
    return f0 + (1.0 - f0) * (1.0 - cos_vh).pow(5)


def ggx_specular(
    normals: Tensor,
    view_directions: Tensor,
    light_directions: Tensor,
    roughness: Tensor,
    f0: Tensor,
) -> Tensor:
    """The Cook-Torrance specular BRDF value, ``D * Vis * F``.

    This is the BRDF itself, not a radiance: multiply by incident radiance and
    ``max(0, n . w_i)`` to get the specular contribution to outgoing radiance,
    exactly as :func:`microfacet_response` does.

    Args:
        normals: ``[N, 3]``, need not be unit.
        view_directions: ``[N, 3]`` unit, pointing away from the surface.
        light_directions: ``[N, 3]`` unit, pointing away from the surface.
        roughness: ``[N]`` perceptual roughness in ``(0, 1]``.
        f0: ``[N]`` or ``[N, 3]`` reflectance at normal incidence.

    Returns:
        ``[N]`` or ``[N, 3]`` (matching ``f0``), non-negative.
    """
    unit_n = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos_nv = (unit_n * view_directions).sum(-1)
    cos_nl = (unit_n * light_directions).sum(-1)
    visible = (cos_nv > 0) & (cos_nl > 0)

    h = half_vector(view_directions, light_directions)
    cos_nh = (unit_n * h).sum(-1)
    cos_vh = (view_directions * h).sum(-1)

    alpha = alpha_from_roughness(roughness)
    d = ggx_distribution(cos_nh, alpha)
    vis = smith_ggx_visibility(cos_nv, cos_nl, alpha)
    value = d * vis
    if f0.ndim > value.ndim:
        value = value.unsqueeze(-1)
        visible = visible.unsqueeze(-1)
    f = fresnel_schlick(cos_vh.unsqueeze(-1) if f0.ndim > cos_vh.ndim else cos_vh, f0)
    return torch.where(visible, value * f, torch.zeros_like(value * f))


def microfacet_response(
    normals: Tensor,
    view_directions: Tensor,
    light_directions: Tensor,
    incident_radiance: Tensor,
    *,
    diffuse_albedo: Tensor,
    specular_f0: Tensor,
    roughness: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Outgoing radiance from a Lambertian-plus-GGX surface under a point light.

    Splits the return into ``(diffuse, specular)`` rather than summing them,
    because the whole reason the synthetic generator wants this is to compare a
    *view-independent* fit (which can only ever reach the diffuse term plus
    whatever it can absorb of the specular one) against a *view-dependent* fit
    that has a real chance at both.

    Args:
        normals: ``[N, 3]``.
        view_directions: ``[N, 3]`` unit, away from the surface.
        light_directions: ``[N, 3]`` unit, away from the surface.
        incident_radiance: ``[N, 3]`` arriving from the light (after falloff).
        diffuse_albedo: ``[N, 3]``.
        specular_f0: ``[N, 3]``.
        roughness: ``[N]``.

    Returns:
        ``(diffuse [N, 3], specular [N, 3])``.
    """
    unit_n = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos_nl = (unit_n * light_directions).sum(-1, keepdim=True).clamp_min(0.0)
    diffuse = (diffuse_albedo / math.pi) * incident_radiance * cos_nl
    brdf = ggx_specular(
        normals, view_directions, light_directions, roughness, specular_f0
    )
    specular = brdf * incident_radiance * cos_nl
    return diffuse, specular
