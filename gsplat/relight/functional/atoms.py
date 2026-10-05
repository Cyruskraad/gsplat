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

"""Light atoms: the basis in which illumination is expressed.

An *atom* is a scalar function on the sphere, :math:`A_k : S^2 \\to \\mathbb{R}`.
The illumination ``E`` enters the renderer only through its projection onto the
atoms,

.. math::
    \\ell_k^c = \\langle A_k, E^c \\rangle = \\int_{S^2} A_k(\\omega) E^c(\\omega)\\, d\\omega

which is ``3 x B`` numbers for the whole scene, computed once per light change.
Everything downstream is linear in ``ell``, which is what makes relighting cost
independent of how complicated the illumination is.

The default atoms are spherical Gaussians,

.. math::
    A_k(\\omega) = \\exp\\big(\\lambda_k (\\omega \\cdot \\xi_k - 1)\\big)

peak-normalised to 1 at the axis, with axes spread by a Fibonacci spiral. Peak
normalisation makes :func:`project_point_light` an interpolation weight in
``[0, 1]``, which is the intuitive reading: a delta light lands on the atoms
that point at it.

Two projections are provided and they must agree, because training sees point
lights and inference sees environments:

- :func:`project_point_light` -- ``E = R * delta(omega - omega_l)``, so the
  integral collapses to ``A_k(omega_l) * R``.
- :func:`project_environment` -- numerical quadrature over an equirectangular
  map with solid-angle weights.

Both are exact statements about the same inner product; no approximation is
introduced by moving between them. What *is* approximate is the representation
of the per-Gaussian transport function in the span of the atoms, and the size of
that error is what :mod:`gsplat.relight.functional.compress` measures.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "fibonacci_sphere",
    "default_sharpness",
    "make_sg_atoms",
    "evaluate_atoms",
    "project_point_light",
    "project_environment",
    "equirect_directions",
    "equirect_solid_angles",
    "atom_gram_matrix",
]


def fibonacci_sphere(
    num_points: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Unit vectors spread near-uniformly over the sphere by a Fibonacci spiral.

    Args:
        num_points: How many directions. Must be positive.

    Returns:
        ``[num_points, 3]`` unit vectors.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")
    idx = torch.arange(num_points, device=device, dtype=dtype)
    z = 1.0 - 2.0 * (idx + 0.5) / num_points
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phi = idx * golden
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def default_sharpness(num_atoms: int) -> float:
    """Sharpness at which ``num_atoms`` Fibonacci lobes just touch.

    Each of ``B`` lobes owns a solid angle of ``4*pi/B``. A spherical cap of
    half-angle ``theta`` subtends ``2*pi*(1 - cos(theta))``, so the lobe's share
    corresponds to ``1 - cos(theta) = 2/B``. Choosing ``lambda`` so that the
    atom has fallen to ``exp(-1)`` at that angle gives ``lambda*(1-cos(theta))
    = 1``.

    Using the small-angle spacing between neighbouring Fibonacci points rather
    than the solid-angle-equivalent cap yields ``lambda = B/(2*pi)``, which is
    the looser of the two and leaves the lobes overlapping instead of leaving
    gaps between them. Gaps in the basis are much worse than overlap -- an
    illumination direction that falls in a gap is simply not representable --
    so the looser value is the default.

    Args:
        num_atoms: Number of atoms ``B``.

    Returns:
        A positive sharpness.
    """
    if num_atoms < 1:
        raise ValueError(f"num_atoms must be >= 1, got {num_atoms}")
    return max(num_atoms / (2.0 * math.pi), 1e-3)


def make_sg_atoms(
    num_atoms: int,
    *,
    sharpness: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """Build a spherical-Gaussian atom basis.

    Args:
        num_atoms: Number of atoms ``B``.
        sharpness: Shared lobe sharpness. Defaults to :func:`default_sharpness`.

    Returns:
        ``(axes, sharpnesses)`` with shapes ``[B, 3]`` and ``[B]``.
    """
    axes = fibonacci_sphere(num_atoms, device=device, dtype=dtype)
    lam = default_sharpness(num_atoms) if sharpness is None else float(sharpness)
    if lam <= 0.0:
        raise ValueError(f"sharpness must be > 0, got {lam}")
    sharpnesses = torch.full((num_atoms,), lam, device=device, dtype=dtype)
    return axes, sharpnesses


def evaluate_atoms(directions: Tensor, axes: Tensor, sharpnesses: Tensor) -> Tensor:
    """Evaluate every atom at every direction.

    Args:
        directions: ``[..., 3]`` unit vectors. Not renormalised -- pass unit
            vectors, because silently normalising here would hide a caller bug
            that matters (a non-unit light direction changes the falloff).
        axes: ``[B, 3]`` atom axes.
        sharpnesses: ``[B]`` atom sharpnesses.

    Returns:
        ``[..., B]`` atom values in ``(0, 1]``.
    """
    if directions.shape[-1] != 3:
        raise ValueError(f"directions must end in 3, got {tuple(directions.shape)}")
    if axes.ndim != 2 or axes.shape[-1] != 3:
        raise ValueError(f"axes must be [B, 3], got {tuple(axes.shape)}")
    if sharpnesses.shape != axes.shape[:1]:
        raise ValueError(
            f"sharpnesses must be [B] matching axes, got "
            f"{tuple(sharpnesses.shape)} vs {tuple(axes.shape)}"
        )
    cos = directions @ axes.transpose(-1, -2)  # [..., B]
    return torch.exp(sharpnesses * (cos - 1.0))


def project_point_light(
    direction: Tensor,
    radiance: Tensor,
    axes: Tensor,
    sharpnesses: Tensor,
) -> Tensor:
    """Project a delta (point) light onto the atoms.

    For ``E(omega) = radiance * delta(omega - direction)`` the inner product
    collapses to ``A_k(direction) * radiance``. This is the projection used
    during training, where every observation is a single light.

    Args:
        direction: ``[..., 3]`` unit vector *towards* the light.
        radiance: ``[..., 3]`` RGB radiance arriving from it.
        axes: ``[B, 3]``.
        sharpnesses: ``[B]``.

    Returns:
        ``[..., 3, B]`` light coefficients.
    """
    if radiance.shape[-1] != 3:
        raise ValueError(f"radiance must end in 3, got {tuple(radiance.shape)}")
    weights = evaluate_atoms(direction, axes, sharpnesses)  # [..., B]
    return radiance.unsqueeze(-1) * weights.unsqueeze(-2)  # [..., 3, B]


def equirect_directions(
    height: int,
    width: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Directions at the centre of every texel of an equirectangular map.

    Row 0 is ``+z`` (theta = 0) and row ``height-1`` is ``-z``; azimuth runs
    over ``[0, 2*pi)`` across the columns.

    Returns:
        ``[height, width, 3]`` unit vectors.
    """
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be >= 1, got {height}x{width}")
    theta = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (
        math.pi / height
    )
    phi = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (
        2.0 * math.pi / width
    )
    sin_t = torch.sin(theta)[:, None]
    cos_t = torch.cos(theta)[:, None]
    return torch.stack(
        [
            sin_t * torch.cos(phi)[None, :],
            sin_t * torch.sin(phi)[None, :],
            cos_t.expand(height, width),
        ],
        dim=-1,
    )


def equirect_solid_angles(
    height: int,
    width: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Solid angle of every texel of an equirectangular map.

    ``dOmega = sin(theta) * dtheta * dphi``. Summing the result gives ``4*pi``
    up to the quadrature error of the midpoint rule, which is what
    ``test_equirect_solid_angles_sum_to_sphere`` pins.

    Returns:
        ``[height, width]`` solid angles.
    """
    theta = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (
        math.pi / height
    )
    d_theta = math.pi / height
    d_phi = 2.0 * math.pi / width
    return (torch.sin(theta) * d_theta * d_phi)[:, None].expand(height, width)


def project_environment(
    envmap: Tensor,
    axes: Tensor,
    sharpnesses: Tensor,
) -> Tensor:
    """Project an equirectangular environment map onto the atoms.

    This is the inference-time projection. It costs ``B * H * W`` multiply-adds
    once per light change and nothing per pixel or per Gaussian.

    Args:
        envmap: ``[H, W, 3]`` linear radiance, laid out to match
            :func:`equirect_directions`.
        axes: ``[B, 3]``.
        sharpnesses: ``[B]``.

    Returns:
        ``[3, B]`` light coefficients.
    """
    if envmap.ndim != 3 or envmap.shape[-1] != 3:
        raise ValueError(f"envmap must be [H, W, 3], got {tuple(envmap.shape)}")
    height, width = envmap.shape[0], envmap.shape[1]
    dirs = equirect_directions(
        height, width, device=envmap.device, dtype=envmap.dtype
    )  # [H, W, 3]
    domega = equirect_solid_angles(
        height, width, device=envmap.device, dtype=envmap.dtype
    )  # [H, W]
    atom_vals = evaluate_atoms(dirs, axes, sharpnesses)  # [H, W, B]
    weighted = atom_vals * domega.unsqueeze(-1)  # [H, W, B]
    return torch.einsum("hwc,hwb->cb", envmap, weighted)


def atom_gram_matrix(
    axes: Tensor,
    sharpnesses: Tensor,
    *,
    resolution: int = 64,
) -> Tensor:
    """Numerical Gram matrix ``<A_j, A_k>`` of the atom basis.

    Diagnostic rather than load-bearing: a Gram matrix far from diagonal means
    the atoms are heavily redundant, and a near-singular one means the basis is
    badly conditioned for the least-squares light solve used by inverse
    lighting.

    Args:
        resolution: Quadrature rows; columns are ``2 * resolution``.

    Returns:
        ``[B, B]``.
    """
    dirs = equirect_directions(
        resolution, 2 * resolution, device=axes.device, dtype=axes.dtype
    )
    domega = equirect_solid_angles(
        resolution, 2 * resolution, device=axes.device, dtype=axes.dtype
    )
    vals = evaluate_atoms(dirs, axes, sharpnesses)  # [H, W, B]
    weighted = vals * domega.unsqueeze(-1)
    return torch.einsum("hwj,hwk->jk", vals, weighted)
