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

"""Roughness prefiltering, moved off the per-frame path.

The sharp, view-dependent part of appearance is evaluated against a
roughness-prefiltered environment. Building that prefiltered pyramid is
normally a per-light-change cost -- the ``olat-relight`` runtime document
budgets 40 ms for it.

Prefiltering is a *linear* operator on the environment, so it commutes with the
atom expansion:

.. math::
    \\mathrm{prefilter}\\Big(\\sum_k \\ell_k A_k\\Big)
        = \\sum_k \\ell_k \\, \\mathrm{prefilter}(A_k)

Prefilter each atom once, at training time. At runtime the prefiltered
environment for *any* illumination is a ``B``-term linear combination of maps
that are already in memory. The 40 ms disappears and what replaces it scales
with ``B`` and the pyramid size, not with the environment's resolution or
content.

The convolution here is direct quadrature rather than an FFT or a mip chain.
It is used offline, once per atom, so clarity beats throughput; and a direct
sum is the thing the linearity test can be trusted against.
"""

from typing import Optional

import torch
from torch import Tensor

from .atoms import equirect_directions, equirect_solid_angles, evaluate_atoms

__all__ = [
    "prefilter_equirect",
    "prefilter_atoms",
    "combine_prefiltered",
    "roughness_to_sharpness",
]


def roughness_to_sharpness(roughness: float) -> float:
    """Map a perceptual roughness in ``(0, 1]`` to a spherical-Gaussian sharpness.

    Uses Disney's ``alpha = roughness^2`` parameterisation and the standard
    spherical-Gaussian fit to a GGX lobe, ``lambda = 2 / alpha^2``. Exact
    agreement with GGX is not claimed and is not needed: the prefiltered
    environment is a band-limiting device, and the fit only has to be
    monotone and smooth in roughness.

    Args:
        roughness: Perceptual roughness in ``(0, 1]``.

    Returns:
        A positive sharpness.
    """
    if not 0.0 < roughness <= 1.0:
        raise ValueError(f"roughness must be in (0, 1], got {roughness}")
    alpha = roughness * roughness
    return 2.0 / (alpha * alpha)


def _convolution_weights(
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    sharpness: float,
    *,
    device: Optional[torch.device],
    dtype: torch.dtype,
) -> Tensor:
    """Normalised spherical convolution weights, ``[out_h*out_w, in_h*in_w]``."""
    src = equirect_directions(in_height, in_width, device=device, dtype=dtype)
    dst = equirect_directions(out_height, out_width, device=device, dtype=dtype)
    domega = equirect_solid_angles(in_height, in_width, device=device, dtype=dtype)
    cos = dst.reshape(-1, 3) @ src.reshape(-1, 3).transpose(0, 1)  # [O, I]
    kernel = torch.exp(sharpness * (cos - 1.0)) * domega.reshape(1, -1)
    return kernel / kernel.sum(dim=-1, keepdim=True).clamp(min=1e-30)


def prefilter_equirect(
    envmap: Tensor,
    sharpness: float,
    *,
    out_height: Optional[int] = None,
    out_width: Optional[int] = None,
) -> Tensor:
    """Convolve an equirectangular map with a spherical-Gaussian kernel.

    Args:
        envmap: ``[H, W, C]`` linear radiance. ``C`` is arbitrary, so this runs
            on a single atom (``C = 1``) exactly as it runs on an RGB
            environment.
        sharpness: Kernel sharpness; larger is sharper.
        out_height: Output rows. Defaults to the input's.
        out_width: Output columns. Defaults to the input's.

    Returns:
        ``[out_height, out_width, C]``.
    """
    if envmap.ndim != 3:
        raise ValueError(f"envmap must be [H, W, C], got {tuple(envmap.shape)}")
    if sharpness <= 0.0:
        raise ValueError(f"sharpness must be > 0, got {sharpness}")
    in_h, in_w, channels = envmap.shape
    out_h = in_h if out_height is None else out_height
    out_w = in_w if out_width is None else out_width
    weights = _convolution_weights(
        in_h, in_w, out_h, out_w, sharpness, device=envmap.device, dtype=envmap.dtype
    )
    flat = envmap.reshape(-1, channels)
    return (weights @ flat).reshape(out_h, out_w, channels)


def prefilter_atoms(
    axes: Tensor,
    sharpnesses: Tensor,
    kernel_sharpness: float,
    *,
    height: int = 16,
    width: int = 32,
    source_height: int = 32,
    source_width: int = 64,
) -> Tensor:
    """Prefilter every atom once, offline.

    Args:
        axes: ``[B, 3]``.
        sharpnesses: ``[B]``.
        kernel_sharpness: From :func:`roughness_to_sharpness`.
        height: Output rows per atom.
        width: Output columns per atom.
        source_height: Quadrature rows used to sample the atoms.
        source_width: Quadrature columns.

    Returns:
        ``[B, height, width]`` prefiltered atom maps.
    """
    dirs = equirect_directions(
        source_height, source_width, device=axes.device, dtype=axes.dtype
    )
    atom_vals = evaluate_atoms(dirs, axes, sharpnesses)  # [Hs, Ws, B]
    filtered = prefilter_equirect(
        atom_vals, kernel_sharpness, out_height=height, out_width=width
    )  # [H, W, B]
    return filtered.permute(2, 0, 1).contiguous()


def combine_prefiltered(atom_maps: Tensor, ell: Tensor) -> Tensor:
    """Build the prefiltered environment for an illumination, at runtime.

    This is the whole per-light-change cost of prefiltering: one ``[B]``-length
    linear combination per texel per channel.

    Args:
        atom_maps: ``[B, H, W]`` from :func:`prefilter_atoms`.
        ell: ``[3, B]`` light coefficients.

    Returns:
        ``[H, W, 3]`` prefiltered environment.
    """
    if atom_maps.ndim != 3:
        raise ValueError(f"atom_maps must be [B, H, W], got {tuple(atom_maps.shape)}")
    if ell.ndim != 2 or ell.shape[0] != 3 or ell.shape[1] != atom_maps.shape[0]:
        raise ValueError(
            f"ell must be [3, {atom_maps.shape[0]}], got {tuple(ell.shape)}"
        )
    return torch.einsum("cb,bhw->hwc", ell, atom_maps)
