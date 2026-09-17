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

"""Measuring how much angular rank a scene's light transport actually has.

Sizing the light basis by taste is the easy mistake. This module sizes it by
measurement instead.

Handheld capture yields unstructured ``(view, light)`` pairs rather than a dense
view-by-light tensor, so the classical eigen-image SVD cannot be taken on the
raw photographs. The fitted transport is dense in the light dimension by
construction, though, so the decomposition is taken there:

1. Train over-complete, with ``B0`` analytic atoms (128 by default).
2. Stack the fitted transport into ``[N*3, B0]`` and take its SVD. By
   Eckart--Young this gives the optimal rank-``B`` approximation under the
   Frobenius norm -- there is no better ``B``-atom basis for this scene.
3. Keep the leading ``B`` directions.

What comes out is a *rotation* of the light basis, not a new function family.
The compressed atoms are linear combinations of the originals,
``A'_j = sum_k R_{j,k} A_k``, so projecting an environment onto them costs one
extra ``[B, B0]`` matrix-vector product and no new quadrature:

    ell' = R @ ell

The singular-value spectrum is the deliverable. It states how much angular rank
the object's transport carries, and it turns "is ``B`` big enough?" from an
argument into a plot with a number on it.
"""

from typing import NamedTuple

import torch
from torch import Tensor

__all__ = [
    "TransportSpectrum",
    "transport_spectrum",
    "energy_retained",
    "rank_for_energy",
    "compress_transport",
    "project_light_to_compressed",
    "CompressedTransport",
]


class TransportSpectrum(NamedTuple):
    """Singular values of the per-primitive transport, largest first."""

    singular_values: Tensor  # [min(N*3, B)]
    num_atoms: int


class CompressedTransport(NamedTuple):
    """A rank-reduced transport together with the basis rotation that made it."""

    transport: Tensor  # [N, 3, B_new]
    rotation: Tensor  # [B_new, B_old], maps old light coefficients to new
    singular_values: Tensor  # [min(N*3, B_old)] of the original, largest first


def _as_matrix(transport: Tensor) -> Tensor:
    if transport.ndim != 3 or transport.shape[1] != 3:
        raise ValueError(f"transport must be [N, 3, B], got {tuple(transport.shape)}")
    return transport.reshape(-1, transport.shape[-1])


def transport_spectrum(transport: Tensor) -> TransportSpectrum:
    """Singular values of ``[N, 3, B]`` transport, flattened to ``[N*3, B]``.

    Args:
        transport: ``[N, 3, B]``.

    Returns:
        A :class:`TransportSpectrum`.
    """
    matrix = _as_matrix(transport)
    values = torch.linalg.svdvals(matrix)
    return TransportSpectrum(singular_values=values, num_atoms=transport.shape[-1])


def energy_retained(singular_values: Tensor, rank: int) -> Tensor:
    """Fraction of squared Frobenius energy kept by a rank-``rank`` truncation.

    Squared because the Frobenius norm is the sum of squared singular values,
    and the truncation error the SVD minimises is measured in that norm. A
    ratio of plain singular values would flatter the truncation.

    Args:
        singular_values: ``[K]`` descending.
        rank: How many to keep, in ``[0, K]``.

    Returns:
        Scalar tensor in ``[0, 1]``.
    """
    if rank < 0 or rank > singular_values.shape[0]:
        raise ValueError(f"rank must be in [0, {singular_values.shape[0]}], got {rank}")
    squared = singular_values.to(torch.float64) ** 2
    total = squared.sum()
    if total <= 0:
        return torch.ones((), dtype=torch.float64, device=singular_values.device)
    return squared[:rank].sum() / total


def rank_for_energy(singular_values: Tensor, threshold: float) -> int:
    """Smallest rank retaining at least ``threshold`` of the squared energy.

    This is the function that picks ``B``.

    Args:
        singular_values: ``[K]`` descending.
        threshold: Target fraction in ``(0, 1]``.

    Returns:
        An integer rank in ``[1, K]``.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    squared = singular_values.to(torch.float64) ** 2
    total = squared.sum()
    if total <= 0:
        return 1
    cumulative = torch.cumsum(squared, dim=0) / total
    hits = torch.nonzero(cumulative >= threshold, as_tuple=False)
    if hits.numel() == 0:
        return int(singular_values.shape[0])
    return int(hits[0, 0].item()) + 1


def compress_transport(transport: Tensor, rank: int) -> CompressedTransport:
    """Optimal rank-``rank`` compression of the light basis.

    Args:
        transport: ``[N, 3, B_old]``.
        rank: Target number of atoms, in ``[1, B_old]``.

    Returns:
        A :class:`CompressedTransport`. Reconstructing
        ``compressed.transport @ compressed.rotation`` recovers the best
        rank-``rank`` approximation of the input.
    """
    matrix = _as_matrix(transport)
    num_old = transport.shape[-1]
    if rank < 1 or rank > num_old:
        raise ValueError(f"rank must be in [1, {num_old}], got {rank}")
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    reduced = u[:, :rank] * s[:rank]  # [N*3, rank]
    rotation = vh[:rank]  # [rank, B_old]
    return CompressedTransport(
        transport=reduced.reshape(transport.shape[0], 3, rank),
        rotation=rotation,
        singular_values=s,
    )


def project_light_to_compressed(ell: Tensor, rotation: Tensor) -> Tensor:
    """Map light coefficients from the original atoms onto the compressed ones.

    Because the compressed atoms are linear combinations of the originals, the
    projection of any environment onto them is the same linear combination of
    the original projections. No new quadrature, no re-derivation, and it holds
    for point lights and environment maps alike.

    Args:
        ell: ``[..., 3, B_old]``.
        rotation: ``[B_new, B_old]`` from :func:`compress_transport`.

    Returns:
        ``[..., 3, B_new]``.
    """
    if rotation.ndim != 2:
        raise ValueError(
            f"rotation must be [B_new, B_old], got {tuple(rotation.shape)}"
        )
    if ell.shape[-1] != rotation.shape[-1]:
        raise ValueError(
            f"ell has {ell.shape[-1]} atoms but rotation expects "
            f"{rotation.shape[-1]}"
        )
    return ell @ rotation.transpose(-1, -2)
