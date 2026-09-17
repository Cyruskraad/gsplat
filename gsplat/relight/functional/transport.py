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

"""Transport contraction: the two render paths, and why they are the same one.

A primitive's outgoing radiance is linear in the illumination by construction:

.. math::
    L_i^c = \\sum_k M_i^{c,k} \\, \\ell^{c,k}

Alpha compositing is a weighted sum whose weights depend only on geometry and
opacity -- never on the light. So for a pixel ``p`` covered by primitives with
compositing weights ``w_i``,

.. math::
    C(p)^c = \\sum_i w_i L_i^c
           = \\sum_i w_i \\sum_k M_i^{c,k} \\ell^{c,k}
           = \\sum_k \\Big( \\sum_i w_i M_i^{c,k} \\Big) \\ell^{c,k}
           = \\sum_k \\mathbb{T}(p)^{c,k} \\ell^{c,k}

The two groupings of that sum are the two render paths:

**Path A, contract then splat.** Evaluate ``L_i = M_i . ell`` per primitive
(:func:`contract`), then rasterise three channels. The contraction does not
depend on the camera, so one contraction serves any number of views -- this is
the path that makes multi-view relighting cheap.

**Path B, splat then contract.** Rasterise the ``3B`` channels of ``M_i``
(:func:`pack_transport`), then contract per pixel (:func:`contract_screen`).
The splat does not depend on the light, so one splat serves any number of
illuminations -- this is the path that makes light editing and inverse lighting
interactive.

This is *not* the usual deferred-shading approximation. Deferred shading
composites normals and roughnesses and then evaluates a non-linear BRDF on the
blend, which is only correct where a pixel is covered by a single opaque
surface. Here the shading operator is linear, so the interchange above is an
identity. :func:`composite` and the tests in ``tests/relight`` exist to hold
that claim to a number rather than to an argument.

Near-field training breaks the *shared* ``ell`` but not the linearity: when
every primitive sees its own light direction, ``ell`` acquires a leading ``N``
and Path A still applies. Path B does not, because there is no single ``ell``
to contract against in screen space. That asymmetry is the honest statement of
what the fast path buys and when.
"""

from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "contract",
    "contract_screen",
    "pack_transport",
    "unpack_transport",
    "composite",
    "compositing_weights",
]


def _check_transport(transport: Tensor) -> Tuple[int, int]:
    if transport.ndim < 3 or transport.shape[-2] != 3:
        raise ValueError(f"transport must be [..., 3, B], got {tuple(transport.shape)}")
    return transport.shape[-2], transport.shape[-1]


def contract(transport: Tensor, ell: Tensor) -> Tensor:
    """Path A: contract per-primitive transport against the light. ``M . ell``.

    Args:
        transport: ``[N, 3, B]`` per-primitive transport.
        ell: ``[3, B]`` for a shared (far-field) illumination, or ``[N, 3, B]``
            when every primitive sees its own light, which is what near-field
            point-light training produces.

    Returns:
        ``[N, 3]`` outgoing radiance.
    """
    _check_transport(transport)
    if transport.ndim != 3:
        raise ValueError(f"transport must be [N, 3, B], got {tuple(transport.shape)}")
    if ell.shape[-2:] != transport.shape[-2:]:
        raise ValueError(
            f"ell trailing dims {tuple(ell.shape[-2:])} must match transport "
            f"{tuple(transport.shape[-2:])}"
        )
    if ell.ndim == 2:
        return torch.einsum("ncb,cb->nc", transport, ell)
    if ell.ndim == 3:
        if ell.shape[0] != transport.shape[0]:
            raise ValueError(
                f"per-primitive ell has {ell.shape[0]} rows but transport has "
                f"{transport.shape[0]}"
            )
        return (transport * ell).sum(dim=-1)
    raise ValueError(f"ell must be [3, B] or [N, 3, B], got {tuple(ell.shape)}")


def contract_screen(transport: Tensor, ell: Tensor) -> Tensor:
    """Path B: contract composited screen-space transport against the light.

    This is the whole cost of changing the illumination once the transport has
    been splatted: ``3B`` multiply-adds per pixel, independent of the number of
    primitives and of how many lights the illumination contains.

    Args:
        transport: ``[..., 3, B]`` composited transport, e.g. ``[H, W, 3, B]``.
        ell: ``[3, B]`` shared light coefficients.

    Returns:
        ``[..., 3]`` radiance.
    """
    _check_transport(transport)
    if ell.ndim != 2 or ell.shape != transport.shape[-2:]:
        raise ValueError(
            f"ell must be [3, B] matching transport trailing dims "
            f"{tuple(transport.shape[-2:])}, got {tuple(ell.shape)}"
        )
    return (transport * ell).sum(dim=-1)


def pack_transport(transport: Tensor) -> Tensor:
    """Flatten ``[..., 3, B]`` transport into ``[..., 3*B]`` splat channels.

    Channel order is channel-major (``c * B + k``), so the first ``B`` entries
    are the red atoms. :func:`unpack_transport` is its exact inverse.
    """
    _check_transport(transport)
    return transport.reshape(*transport.shape[:-2], -1)


def unpack_transport(packed: Tensor, num_atoms: int) -> Tensor:
    """Inverse of :func:`pack_transport`."""
    if num_atoms < 1:
        raise ValueError(f"num_atoms must be >= 1, got {num_atoms}")
    if packed.shape[-1] != 3 * num_atoms:
        raise ValueError(
            f"packed last dim {packed.shape[-1]} is not 3 * num_atoms "
            f"({3 * num_atoms})"
        )
    return packed.reshape(*packed.shape[:-1], 3, num_atoms)


def compositing_weights(alphas: Tensor) -> Tensor:
    """Front-to-back alpha compositing weights ``w_i = a_i * prod_{j<i}(1-a_j)``.

    Args:
        alphas: ``[..., P]`` per-primitive alphas, already sorted front to back.

    Returns:
        ``[..., P]`` weights. They sum to ``1 - prod_i (1 - a_i) <= 1``; the
        shortfall is the background transmittance.
    """
    if alphas.shape[-1] < 1:
        raise ValueError("alphas must have at least one primitive")
    one_minus = 1.0 - alphas
    # Exclusive cumulative product, built by shifting rather than by dividing
    # the inclusive product. Dividing is the obvious trick and it is wrong
    # exactly where it matters: a fully opaque primitive makes the divisor zero,
    # and every weight in front of it -- including its own -- collapses to zero.
    inclusive = torch.cumprod(one_minus, dim=-1)
    transmittance = torch.cat(
        [torch.ones_like(inclusive[..., :1]), inclusive[..., :-1]], dim=-1
    )
    return alphas * transmittance


def composite(alphas: Tensor, features: Tensor) -> Tensor:
    """Alpha-composite arbitrary per-primitive features, front to back.

    Deliberately free of any rasteriser: the exactness claim is a property of
    compositing itself, so proving it must not depend on CUDA, on a tile
    schedule, or on anything that only runs on a workstation.

    Args:
        alphas: ``[..., P]`` sorted front to back.
        features: ``[..., P, C]`` per-primitive features.

    Returns:
        ``[..., C]`` composited features.
    """
    if features.shape[:-1] != alphas.shape:
        raise ValueError(
            f"features leading dims {tuple(features.shape[:-1])} must match "
            f"alphas {tuple(alphas.shape)}"
        )
    weights = compositing_weights(alphas)
    return (weights.unsqueeze(-1) * features).sum(dim=-2)
