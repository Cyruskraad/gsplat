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

"""View-dependent gloss that costs the illumination-independence claim nothing.

The representation so far has no view dependence: ``L_i = M_i . ell`` is a
function of the incident illumination alone. That is exact for a Lambertian
surface and wrong for a glossy one, and on a flash-on-camera capture it is also
*unfalsifiable*, because every observation lies on the retroreflective diagonal
where incident and outgoing directions coincide.

The fix rests on one line of the exactness theorem in
:mod:`atlas.functional.transport`:

    compositing weights depend only on geometry and opacity -- never on the light

It constrains ``M_i`` to be independent of **the light**. It says nothing about
the **view**. So a view-dependent ``M_i(w_o)`` preserves the theorem, Path A /
Path B equivalence, and the property the project exists for: relighting costs
the same whatever the illumination contains.

## The construction

Prefiltering is a linear operator on the environment, so it commutes with the
atom expansion -- :mod:`atlas.functional.prefilter` proves this and a test pins
it. Prefilter every atom once, offline, at a ladder of roughness levels. Then

.. math::
    L_i(w_o) = \\underbrace{\\sum_k \\ell_k M_{i,k}}_{\\text{diffuse}}
             + \\underbrace{\\rho_{s,i} \\sum_k \\ell_k
               \\hat{A}_k\\big(r_i(w_o); \\alpha_i\\big)}_{\\text{specular}}

where :math:`\\hat{A}_k(\\cdot; \\alpha)` is atom *k* prefiltered at roughness
:math:`\\alpha` and sampled along the reflection direction. Everything inside
the second sum depends on the view and the primitive; nothing depends on
``ell``. Radiance stays linear in the illumination.

It is also *cheaper* than the diffuse term. The specular weights are ``[N, B]``
rather than ``[N, 3, B]`` -- one lobe shared across colour channels, with the
colour carried by :math:`\\rho_s` -- so the contraction is an ``nk,ck->nc``
einsum with no ``[N, 3, B]`` intermediate at all.

## Why this is learnable from a co-located capture

For a microfacet BRDF the half-vector is ``normalize(w_i + w_o)``; when the two
coincide it is simply ``w``. The observed specular therefore samples the normal
distribution at ``D(n . w)``, and an orbit sweeps ``n . w`` through its whole
range. Roughness is identifiable from a flash-on-camera pass -- this is the
setting of Nam et al., *Practical SVBRDF Acquisition with Unstructured Flash
Photography* (2018). The parameter is constrained by the data, not guessed.

Co-located capture also removes a problem: shadow rays coincide with view rays,
so nothing visible is ever in shadow and no shadow term is needed.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

import torch
from torch import Tensor

from .calibration import reflect as mirror
from .prefilter import (
    prefilter_atoms,
    roughness_to_sharpness,
)

__all__ = [
    "RoughnessLadder",
    "build_roughness_ladder",
    "sample_equirect",
    "direction_to_texel",
    "reflect",
    "specular_weights",
    "DEFAULT_ROUGHNESS_LEVELS",
    "DEFAULT_LADDER_HEIGHT",
    "DEFAULT_LADDER_WIDTH",
    "min_resolvable_roughness",
]

#: Roughness levels the ladder is built at. Geometric rather than uniform: the
#: prefiltered lobe changes far faster at the sharp end, so equal *ratios* keep
#: the interpolation error roughly even across the range. Every one of these is
#: resolvable at :data:`DEFAULT_LADDER_HEIGHT` -- see
#: :func:`min_resolvable_roughness` for why that is a real constraint.
DEFAULT_ROUGHNESS_LEVELS = (0.30, 0.45, 0.65, 1.0)

#: Default resolution of each prefiltered map.
DEFAULT_LADDER_HEIGHT = 64
DEFAULT_LADDER_WIDTH = 128


def min_resolvable_roughness(height: int) -> float:
    """The sharpest gloss an equirect map of this height can actually hold.

    A prefiltered map band-limits by construction: it cannot represent a lobe
    narrower than one texel. A spherical Gaussian of sharpness ``lambda`` has
    angular width about ``1/sqrt(lambda)``, a texel spans ``pi/height``, and
    the Disney fit used here is ``lambda = 2 / roughness^4``. Putting those
    together, the sharpest representable roughness is
    ``(2 * (pi / height)^2)^(1/4)``.

    This matters more than it sounds. Asking for roughness 0.05 at a 32-row map
    means ``lambda = 320000``, a lobe some four thousand times narrower than a
    texel -- and the quadrature returns **5.6e-142**, a silent collapse to zero
    that renders as a missing highlight rather than as an error. So
    :func:`build_roughness_ladder` refuses it and says what resolution would be
    needed instead.

    At 64 rows the limit is roughness 0.26; at 256 rows it is 0.16. Mirror-like
    surfaces are out of range for this representation, which is honest: a
    co-located capture cannot determine them either.
    """
    if height < 1:
        raise ValueError(f"height must be at least 1, got {height}")
    return float((2.0 * (math.pi / height) ** 2) ** 0.25)


class RoughnessLadder(NamedTuple):
    """Prefiltered atoms at a few roughness levels, built once.

    Attributes:
        maps: ``[L, B, H, W]`` -- level, atom, equirect row, equirect column.
        levels: ``[L]`` roughness each level was filtered at, ascending.
    """

    maps: Tensor
    levels: Tensor

    @property
    def num_atoms(self) -> int:
        return int(self.maps.shape[1])

    def to(self, *args, **kwargs) -> "RoughnessLadder":
        return RoughnessLadder(
            self.maps.to(*args, **kwargs), self.levels.to(*args, **kwargs)
        )


def build_roughness_ladder(
    axes: Tensor,
    sharpnesses: Tensor,
    *,
    levels: Sequence[float] = DEFAULT_ROUGHNESS_LEVELS,
    height: int = DEFAULT_LADDER_HEIGHT,
    width: int = DEFAULT_LADDER_WIDTH,
    source_height: Optional[int] = None,
    source_width: Optional[int] = None,
) -> RoughnessLadder:
    """Prefilter every atom at every roughness level. Offline, once.

    This is the whole cost of the specular term's illumination independence.
    It depends on the basis, not on the scene, the view or the light, so it is
    rebuilt only when the atoms move -- which, with learned atoms, means once
    every few hundred steps rather than once per frame.

    Args:
        axes: ``[B, 3]`` atom axes.
        sharpnesses: ``[B]``.
        levels: Roughness values, ascending.
        height, width: Resolution of each prefiltered map.
        source_height, source_width: Quadrature resolution used to filter.

    Returns:
        A :class:`RoughnessLadder`.
    """
    ordered = list(levels)
    if not ordered:
        raise ValueError("the ladder needs at least one roughness level")
    if any(r <= 0.0 or r > 1.0 for r in ordered):
        raise ValueError(f"roughness levels must lie in (0, 1], got {ordered}")
    if ordered != sorted(ordered):
        raise ValueError(f"roughness levels must be ascending, got {ordered}")

    floor = min_resolvable_roughness(height)
    if ordered[0] < floor:
        needed = int(math.ceil(math.pi * math.sqrt(2.0) / ordered[0] ** 2))
        raise ValueError(
            f"roughness {ordered[0]} needs a lobe narrower than one texel of a "
            f"{height}-row map, which quadrature returns as zero rather than as "
            f"an error. The sharpest this resolution can hold is "
            f"{floor:.3f}; {ordered[0]} would need about {needed} rows."
        )
    if source_height is None:
        source_height = 2 * height
    if source_width is None:
        source_width = 2 * width

    maps = torch.stack(
        [
            prefilter_atoms(
                axes,
                sharpnesses,
                roughness_to_sharpness(level),
                height=height,
                width=width,
                source_height=source_height,
                source_width=source_width,
            )
            for level in ordered
        ]
    )
    return RoughnessLadder(
        maps=maps,
        levels=torch.tensor(ordered, dtype=maps.dtype, device=maps.device),
    )


# --- sampling ---------------------------------------------------------------


def direction_to_texel(directions: Tensor, height: int, width: int):
    """Continuous ``(row, column)`` for unit directions, in texel coordinates.

    Matches :func:`~atlas.functional.atoms.equirect_directions` exactly: row 0
    is ``+z``, azimuth runs over ``[0, 2*pi)``, and both are sampled at texel
    *centres*, so the offset of half a texel is part of the convention rather
    than an off-by-one.
    """
    if directions.shape[-1] != 3:
        raise ValueError(f"directions must end in 3, got {tuple(directions.shape)}")
    unit = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    theta = torch.acos(unit[..., 2].clamp(-1.0, 1.0))
    phi = torch.atan2(unit[..., 1], unit[..., 0]) % (2.0 * math.pi)
    row = theta * (height / math.pi) - 0.5
    column = phi * (width / (2.0 * math.pi)) - 0.5
    return row, column


def sample_equirect(maps: Tensor, directions: Tensor) -> Tensor:
    """Bilinearly sample ``[C, H, W]`` equirect maps along ``[N, 3]`` directions.

    Columns wrap, because azimuth does. Rows clamp, because latitude does not:
    a direction past the pole is the pole.

    Returns:
        ``[N, C]``.
    """
    if maps.ndim != 3:
        raise ValueError(f"maps must be [C, H, W], got {tuple(maps.shape)}")
    channels, height, width = maps.shape
    row, column = direction_to_texel(directions, height, width)

    row0 = torch.floor(row)
    col0 = torch.floor(column)
    dr = (row - row0).unsqueeze(-1)
    dc = (column - col0).unsqueeze(-1)

    r0 = row0.long().clamp(0, height - 1)
    r1 = (row0.long() + 1).clamp(0, height - 1)
    c0 = col0.long() % width
    c1 = (col0.long() + 1) % width

    flat = maps.reshape(channels, height * width)

    def gather(rows: Tensor, cols: Tensor) -> Tensor:
        return flat[:, rows * width + cols].transpose(0, 1)  # [N, C]

    top = gather(r0, c0) * (1 - dc) + gather(r0, c1) * dc
    bottom = gather(r1, c0) * (1 - dc) + gather(r1, c1) * dc
    return top * (1 - dr) + bottom * dr


def reflect(view_directions: Tensor, normals: Tensor) -> Tensor:
    """Unit mirror direction, normalising first.

    The reflection itself is :func:`atlas.functional.calibration.reflect`, which
    the chrome-sphere calibration already needed and which assumes unit inputs.
    Callers here do not have them -- ``camera - mean`` is a displacement -- so
    this normalises, delegates, and normalises the result.

    Args:
        view_directions: ``[N, 3]`` pointing **from** the primitive **towards**
            the camera, the same sense the light directions use.
        normals: ``[N, 3]``, not required to be unit.

    Returns:
        ``[N, 3]`` unit reflection directions.
    """
    if view_directions.shape != normals.shape:
        raise ValueError(
            f"view_directions {tuple(view_directions.shape)} and normals "
            f"{tuple(normals.shape)} must match"
        )
    unit_n = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit_v = view_directions / view_directions.norm(dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    reflected = mirror(unit_v, unit_n)
    return reflected / reflected.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def specular_weights(
    ladder: RoughnessLadder,
    directions: Tensor,
    roughness: Tensor,
) -> Tensor:
    """Per-primitive atom weights for the specular lobe. ``[N, B]``.

    Interpolates between the two ladder levels bracketing each primitive's
    roughness -- the split-sum construction, as real-time image-based lighting
    does it. Blending in roughness rather than rebuilding the filter per
    primitive is what keeps this a lookup.

    Args:
        ladder: From :func:`build_roughness_ladder`.
        directions: ``[N, 3]``, usually the reflection direction.
        roughness: ``[N]`` in ``(0, 1]``; clamped to the ladder's range.

    Returns:
        ``[N, B]``, independent of the illumination by construction.
    """
    if roughness.ndim != 1 or roughness.shape[0] != directions.shape[0]:
        raise ValueError(
            f"roughness must be [{directions.shape[0]}], got {tuple(roughness.shape)}"
        )
    levels = ladder.levels.to(directions.dtype)
    clamped = roughness.to(directions.dtype).clamp(float(levels[0]), float(levels[-1]))

    upper = torch.searchsorted(levels, clamped.contiguous()).clamp(1, len(levels) - 1)
    lower = upper - 1
    span = (levels[upper] - levels[lower]).clamp_min(1e-12)
    blend = ((clamped - levels[lower]) / span).unsqueeze(-1)  # [N, 1]

    maps = ladder.maps.to(directions.dtype)
    # One sample per level, then gather the two each primitive needs. The ladder
    # has a handful of levels, so sampling all of them and selecting is cheaper
    # and far simpler than a gather over a ragged index.
    per_level = torch.stack([sample_equirect(level, directions) for level in maps])
    index = torch.arange(directions.shape[0], device=directions.device)
    return per_level[lower, index] * (1 - blend) + per_level[upper, index] * blend
