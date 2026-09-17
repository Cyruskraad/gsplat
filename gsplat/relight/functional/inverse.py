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

"""Inverse lighting: solve for the illumination that explains an image.

Because the renderer is linear in the light, ``C = T ell``, recovering the
illumination from a target image is a least-squares problem in ``B`` unknowns
per channel rather than an optimisation through a renderer:

.. math::
    \\ell^* = \\arg\\min_{\\ell \\succeq 0} \\; \\| \\mathbb{T}\\,\\ell - I^* \\|^2

The pixels enter only through the normal equations ``G = T^T T`` (``[B, B]``)
and ``c = T^T I`` (``[B]``), accumulated in one pass. Everything after that is
``B``-dimensional and independent of image size, which is why this is a control
rather than a batch job.

Non-negativity is not cosmetic: without it the solver happily returns negative
radiance in directions the image does not constrain, and the resulting
environment looks correct on the view it was fitted to and wrong everywhere
else. It is enforced by projected gradient with Nesterov acceleration on the
normal equations -- no SciPy, no new dependency, and the small dense problem
converges in tens of iterations.
"""

from typing import NamedTuple, Optional

import torch
from torch import Tensor

__all__ = ["LightSolution", "normal_equations", "solve_light"]


class LightSolution(NamedTuple):
    """Recovered light coefficients and what the solver did to get them."""

    ell: Tensor  # [3, B]
    iterations: int
    residual: Tensor  # scalar, ||T ell - I||^2 summed over pixels and channels


def normal_equations(
    transport: Tensor,
    target: Tensor,
    *,
    weights: Optional[Tensor] = None,
) -> tuple:
    """Accumulate ``(G, c)`` per channel from screen-space transport and a target.

    Args:
        transport: ``[P, 3, B]`` composited transport for ``P`` pixels.
        target: ``[P, 3]`` observed linear radiance.
        weights: ``[P]`` optional non-negative per-pixel weights, e.g. a mask.

    Returns:
        ``(gram [3, B, B], rhs [3, B])``.
    """
    if transport.ndim != 3 or transport.shape[1] != 3:
        raise ValueError(f"transport must be [P, 3, B], got {tuple(transport.shape)}")
    if target.shape != transport.shape[:2]:
        raise ValueError(
            f"target must be [P, 3] matching transport, got {tuple(target.shape)}"
        )
    t = transport.permute(1, 0, 2)  # [3, P, B]
    b = target.transpose(0, 1)  # [3, P]
    if weights is not None:
        if weights.shape != transport.shape[:1]:
            raise ValueError(f"weights must be [P], got {tuple(weights.shape)}")
        if bool((weights < 0).any()):
            raise ValueError("weights must be non-negative")
        w = weights.unsqueeze(0).unsqueeze(-1)  # [1, P, 1]
        tw = t * w
    else:
        tw = t
    gram = torch.einsum("cpj,cpk->cjk", tw, t)
    rhs = torch.einsum("cpj,cp->cj", tw, b)
    return gram, rhs


def solve_light(
    transport: Tensor,
    target: Tensor,
    *,
    weights: Optional[Tensor] = None,
    non_negative: bool = True,
    ridge: float = 1e-8,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> LightSolution:
    """Recover ``ell`` from a screen-space transport buffer and a target image.

    Args:
        transport: ``[P, 3, B]`` composited transport.
        target: ``[P, 3]`` observed linear radiance.
        weights: ``[P]`` optional per-pixel weights.
        non_negative: Constrain ``ell >= 0``. Turning this off makes the solve a
            single linear system and is useful only for diagnosing conditioning.
        ridge: Tikhonov term added to the Gram diagonal. The Gram matrix is
            genuinely rank-deficient whenever the view does not constrain every
            atom, which is the normal case, so this is a correctness measure
            rather than a numerical nicety.
        max_iterations: Cap on projected-gradient steps.
        tolerance: Stop when the projected gradient -- the first-order
            optimality measure for the constrained problem -- falls below this
            fraction of the right-hand side's scale.

    Returns:
        A :class:`LightSolution`.
    """
    if ridge < 0.0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    gram, rhs = normal_equations(transport, target, weights=weights)
    num_atoms = gram.shape[-1]
    eye = torch.eye(num_atoms, device=gram.device, dtype=gram.dtype)
    gram = gram + ridge * eye

    if not non_negative:
        ell = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
        return LightSolution(
            ell=ell,
            iterations=0,
            residual=_residual(transport, target, ell, weights),
        )

    # Lipschitz constant of the gradient, per channel: the largest eigenvalue of
    # the (symmetric positive semi-definite) Gram matrix.
    lipschitz = torch.linalg.eigvalsh(gram)[:, -1].clamp(min=1e-20)  # [3]
    step = (1.0 / lipschitz).unsqueeze(-1)  # [3, 1]
    scale = float(torch.linalg.norm(rhs).clamp(min=1.0))

    ell = torch.zeros_like(rhs)
    momentum = ell.clone()
    t_k = 1.0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        grad = torch.einsum("cjk,ck->cj", gram, momentum) - rhs
        nxt = torch.clamp(momentum - step * grad, min=0.0)

        # Adaptive gradient restart. Plain FISTA ripples badly once the Gram
        # matrix is ill-conditioned, which it always is here -- a single view
        # never constrains every atom. Restarting whenever the momentum points
        # uphill costs one inner product and removes the ripple.
        if float(((momentum - nxt) * (nxt - ell)).sum()) > 0.0:
            t_k = 1.0
            momentum = nxt.clone()
        else:
            t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_k * t_k) ** 0.5)
            momentum = nxt + ((t_k - 1.0) / t_next) * (nxt - ell)
            t_k = t_next
        ell = nxt

        # Stop on first-order optimality, not on step size. For a non-negatively
        # constrained problem the KKT residual is the gradient where the
        # variable is free and its negative part where the variable sits on the
        # bound; a step-size rule instead reports "not converged" forever while
        # the objective has long since stopped moving.
        grad_at_ell = torch.einsum("cjk,ck->cj", gram, ell) - rhs
        projected = torch.where(ell > 0, grad_at_ell, torch.clamp(grad_at_ell, max=0.0))
        if float(torch.linalg.norm(projected)) <= tolerance * scale:
            break

    return LightSolution(
        ell=ell,
        iterations=iterations,
        residual=_residual(transport, target, ell, weights),
    )


def _residual(
    transport: Tensor,
    target: Tensor,
    ell: Tensor,
    weights: Optional[Tensor],
) -> Tensor:
    predicted = (transport * ell.unsqueeze(0)).sum(dim=-1)  # [P, 3]
    error = (predicted - target) ** 2
    if weights is not None:
        error = error * weights.unsqueeze(-1)
    return error.sum()
