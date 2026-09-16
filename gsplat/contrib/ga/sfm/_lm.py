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
"""Schur-complement Levenberg-Marquardt, independent of pose representation.

Both bundle-adjustment arms -- the motor one in
:mod:`gsplat.contrib.ga.sfm.ba` and the quaternion/se(3) control in
:mod:`gsplat.contrib.ga.baseline.ba` -- run *this* solver. That is deliberate.
The research question is what the pose parameterization buys, so the
parameterization has to be the only thing that differs between the arms; a
second solver would confound the comparison with its own damping schedule,
stopping rule and linear algebra.

The caller supplies three callbacks over an opaque state:

- ``residual_and_jacobians(state) -> (r, J_cam, J_pt)`` shaped ``(M, 2)``,
  ``(M, 2, 6)``, ``(M, 2, 3)``
- ``apply_step(state, delta_cam, delta_pt) -> state``
- ``cost(state) -> scalar``

The point block of the normal equations is block-diagonal, so it inverts
per-point and can be eliminated, leaving a system in the cameras alone.
Forming the full Jacobian densely instead would be hundreds of megabytes at
even modest problem sizes.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

__all__ = ["schur_lm"]

_EPS = 1e-12


def schur_lm(
    state: Any,
    residual_and_jacobians: Callable[[Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    apply_step: Callable[[Any, torch.Tensor, torch.Tensor], Any],
    cost: Callable[[Any], torch.Tensor],
    num_cameras: int,
    num_points: int,
    camera_idx: torch.Tensor,
    point_idx: torch.Tensor,
    iterations: int = 30,
    fixed_cameras: tuple[int, ...] = (0,),
    damping: float = 1e-4,
    tolerance: float = 1e-12,
) -> tuple[Any, dict]:
    """Minimize ``cost`` over cameras and points. Returns ``(state, stats)``.

    Steps that do not reduce the cost are rejected and the damping is raised,
    so the reported cost history is monotone by construction.
    """
    probe, _, _ = residual_and_jacobians(state)
    dtype, device = probe.dtype, probe.device

    fixed = torch.zeros(num_cameras, dtype=torch.bool, device=device)
    for index in fixed_cameras:
        fixed[index] = True
    gauge_mask = fixed.repeat_interleave(6)

    eye3 = torch.eye(3, dtype=dtype, device=device)
    current = cost(state)
    history = [float(current)]
    lam = damping

    for _ in range(iterations):
        residual, jac_cam, jac_pt = residual_and_jacobians(state)

        u_blocks = torch.zeros(num_cameras, 6, 6, dtype=dtype, device=device)
        v_blocks = torch.zeros(num_points, 3, 3, dtype=dtype, device=device)
        g_cam = torch.zeros(num_cameras, 6, dtype=dtype, device=device)
        g_pt = torch.zeros(num_points, 3, dtype=dtype, device=device)

        u_blocks.index_add_(0, camera_idx, jac_cam.transpose(-2, -1) @ jac_cam)
        v_blocks.index_add_(0, point_idx, jac_pt.transpose(-2, -1) @ jac_pt)
        g_cam.index_add_(
            0, camera_idx, (jac_cam.transpose(-2, -1) @ residual.unsqueeze(-1)).squeeze(-1)
        )
        g_pt.index_add_(
            0, point_idx, (jac_pt.transpose(-2, -1) @ residual.unsqueeze(-1)).squeeze(-1)
        )

        # The camera-point coupling block, scattered into (V, 6, P, 3).
        contributions = jac_cam.transpose(-2, -1) @ jac_pt
        flat = torch.zeros(num_cameras * num_points, 6, 3, dtype=dtype, device=device)
        flat.index_add_(0, camera_idx * num_points + point_idx, contributions)
        w_blocks = flat.reshape(num_cameras, num_points, 6, 3).permute(0, 2, 1, 3)

        accepted = False
        for _ in range(12):
            u_damped = u_blocks + lam * torch.diag_embed(
                torch.diagonal(u_blocks, dim1=-2, dim2=-1)
            )
            v_damped = (
                v_blocks
                + lam * torch.diag_embed(torch.diagonal(v_blocks, dim1=-2, dim2=-1))
                + 1e-9 * eye3
            )
            v_inv = torch.linalg.pinv(v_damped)

            y = torch.einsum("vipj,pjk->vipk", w_blocks, v_inv)
            reduced = torch.block_diag(*u_damped) - torch.einsum(
                "vipj,wkpj->viwk", y, w_blocks
            ).reshape(num_cameras * 6, num_cameras * 6)
            rhs = -(g_cam - torch.einsum("vipj,pj->vi", y, g_pt)).reshape(num_cameras * 6)

            # Gauge fixing: bundle adjustment is invariant to a global
            # similarity, so without pinning a camera the system is singular.
            reduced[gauge_mask, :] = 0.0
            reduced[:, gauge_mask] = 0.0
            reduced[gauge_mask, gauge_mask] = 1.0
            rhs[gauge_mask] = 0.0

            try:
                delta_cam = torch.linalg.solve(reduced, rhs.unsqueeze(-1)).squeeze(-1)
            except Exception:
                lam *= 10.0
                continue
            if not torch.isfinite(delta_cam).all():
                lam *= 10.0
                continue
            delta_cam = delta_cam.reshape(num_cameras, 6)

            back = torch.einsum("vipj,vi->pj", w_blocks, delta_cam)
            delta_pt = -torch.einsum("pjk,pk->pj", v_inv, g_pt + back)

            candidate = apply_step(state, delta_cam, delta_pt)
            new_cost = cost(candidate)
            if torch.isfinite(new_cost) and new_cost < current:
                state, current = candidate, new_cost
                lam = max(lam * 0.3, 1e-12)
                history.append(float(current))
                accepted = True
                break
            lam *= 10.0

        if not accepted or (len(history) > 1 and history[-2] - history[-1] < tolerance):
            break

    return state, {
        "cost": history,
        "final_cost": float(current),
        "iterations": len(history) - 1,
        "damping": lam,
    }
