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
"""Triangulation by incidence.

A pixel back-projects to a *line*, and a 3D point that explains a set of
observations is the point most nearly incident on all of those lines. That
sentence is the whole algorithm, and in PGA it is also the implementation.

Two solvers:

- :func:`triangulate_linear` uses the two planes whose meet is each ray. Point-
  on-plane is a linear constraint, so stacking them gives an ordinary least
  squares problem -- the DLT system, reached through incidence rather than
  through a hand-assembled measurement matrix.
- :func:`triangulate_midpoint` minimizes the geometric point-to-ray distance
  instead. Worth being precise about why this is *also* a linear solve: the
  point-line residual is the normal of the plane joining point to ray, which is
  linear in the point, so its normal equations do not depend on the current
  estimate and there is nothing to iterate. Calling it "nonlinear refinement"
  would be a misnomer for a direct solve.
- :func:`triangulate_reprojection` is the genuinely nonlinear one: it minimizes
  reprojection error in pixels, which is the right objective under Gaussian
  pixel noise, and is the only one of the three whose Jacobian depends on the
  point being estimated.

All three are batched over points and differentiable, so any of them can sit
inside a larger optimization.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import camera as _cam
from gsplat.contrib.ga import motor as _mot
from gsplat.contrib.ga import primitives as _prim

__all__ = [
    "triangulate_linear",
    "triangulate_midpoint",
    "triangulate_reprojection",
    "ray_residuals",
    "reprojection_residuals",
]

_EPS = 1e-12


def _stack_ray_planes(
    motors: torch.Tensor, intrinsics: torch.Tensor, pixels: torch.Tensor
) -> torch.Tensor:
    """Per-observation ray planes, shaped ``(N, 2V, 4)``."""
    views = motors.shape[0]
    num = pixels.shape[1]
    motors_e = motors[:, None, :].expand(views, num, 8)
    intr_e = intrinsics[:, None, :, :].expand(views, num, 3, 3)
    plane_u, plane_v = _cam.pixel_ray_planes(motors_e, intr_e, pixels)
    # (V, N, 4) x2 -> (N, 2V, 4)
    return torch.cat([plane_u, plane_v], dim=0).permute(1, 0, 2)


def triangulate_linear(
    motors: torch.Tensor,
    intrinsics: torch.Tensor,
    pixels: torch.Tensor,
    mask: torch.Tensor | None = None,
    min_singular_value: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares triangulation from ray planes.

    Args:
        motors: camera-from-world motors ``(V, 8)``.
        intrinsics: pinhole ``K`` per view ``(V, 3, 3)``.
        pixels: observations ``(V, N, 2)``.
        mask: ``(V, N)`` bool, which observations exist. Missing observations
            are zero-weighted rather than dropped, so every point keeps the same
            system shape and the solve stays batched.
        min_singular_value: conditioning floor on the ``3x3`` normal matrix.
            Points below it are geometrically undetermined -- a single view, or
            rays that are all but parallel -- and are reported invalid instead
            of returned as a plausible-looking result.

    Returns:
        ``(points, valid)`` of shapes ``(N, 3)`` and ``(N,)``.
    """
    views, num, _ = pixels.shape
    planes = _stack_ray_planes(motors, intrinsics, pixels)  # (N, 2V, 4)
    normals, offsets = planes[..., :3], planes[..., 3]

    if mask is None:
        weights = torch.ones(num, 2 * views, dtype=pixels.dtype, device=pixels.device)
    else:
        weights = torch.cat([mask, mask], dim=0).permute(1, 0).to(pixels.dtype)

    weighted = normals * weights.unsqueeze(-1)
    normal_matrix = weighted.transpose(-2, -1) @ normals  # (N, 3, 3)
    rhs = -(weighted * offsets.unsqueeze(-1)).sum(dim=-2)  # (N, 3)

    # SVD rather than a plain solve: it yields the conditioning that decides
    # validity, and stays finite on the degenerate systems it is diagnosing.
    u, s, vh = torch.linalg.svd(normal_matrix)
    valid = s[..., -1] > min_singular_value
    inv_s = torch.where(s > min_singular_value, 1.0 / s.clamp_min(_EPS), torch.zeros_like(s))
    pseudo_inverse = vh.transpose(-2, -1) @ (inv_s.unsqueeze(-1) * u.transpose(-2, -1))
    points = (pseudo_inverse @ rhs.unsqueeze(-1)).squeeze(-1)
    return points, valid


def ray_residuals(
    points: torch.Tensor,
    motors: torch.Tensor,
    intrinsics: torch.Tensor,
    pixels: torch.Tensor,
) -> torch.Tensor:
    """Point-to-ray residual vectors ``(V, N, 3)``; their norms are distances."""
    views, num, _ = pixels.shape
    motors_e = motors[:, None, :].expand(views, num, 8)
    intr_e = intrinsics[:, None, :, :].expand(views, num, 3, 3)
    rays = _cam.pixel_ray(motors_e, intr_e, pixels)
    return _prim.point_line_residual(points.expand(views, num, 3), rays)


def triangulate_midpoint(
    motors: torch.Tensor,
    intrinsics: torch.Tensor,
    pixels: torch.Tensor,
    mask: torch.Tensor | None = None,
    damping: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The point minimizing summed squared distance to the observation rays.

    Despite being the "geometric" criterion, this is a direct linear solve, not
    an iteration: the distance from a point to a line is ``(I - d d^T)(x - o)``,
    linear in ``x``, so the normal equations are independent of the estimate.
    """
    views, num, _ = pixels.shape
    motors_e = motors[:, None, :].expand(views, num, 8)
    intr_e = intrinsics[:, None, :, :].expand(views, num, 3, 3)
    rays = _prim.normalize_line(_cam.pixel_ray(motors_e, intr_e, pixels))
    directions = _prim.line_direction(rays)
    origins = torch.linalg.cross(directions, _prim.line_moment(rays), dim=-1)

    weights = (
        torch.ones(views, num, dtype=pixels.dtype, device=pixels.device)
        if mask is None
        else mask.to(pixels.dtype)
    )
    eye = torch.eye(3, dtype=pixels.dtype, device=pixels.device)
    projector = eye - directions.unsqueeze(-1) * directions.unsqueeze(-2)
    weighted = projector * weights[..., None, None]

    lhs = weighted.sum(dim=0)
    rhs = (weighted @ origins.unsqueeze(-1)).squeeze(-1).sum(dim=0)
    u, s, vh = torch.linalg.svd(lhs + damping * eye)
    valid = s[..., -1] > 1e-6
    inv_s = torch.where(s > 1e-6, 1.0 / s.clamp_min(_EPS), torch.zeros_like(s))
    pinv = vh.transpose(-2, -1) @ (inv_s.unsqueeze(-1) * u.transpose(-2, -1))
    return (pinv @ rhs.unsqueeze(-1)).squeeze(-1), valid


def reprojection_residuals(
    points: torch.Tensor,
    motors: torch.Tensor,
    intrinsics: torch.Tensor,
    pixels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reprojection residuals ``(V, N, 2)`` in pixels, with a validity mask."""
    views, num, _ = pixels.shape
    motors_e = motors[:, None, :].expand(views, num, 8)
    intr_e = intrinsics[:, None, :, :].expand(views, num, 3, 3)
    projected, valid = _cam.project(motors_e, intr_e, points.expand(views, num, 3))
    return projected - pixels, valid


def triangulate_reprojection(
    motors: torch.Tensor,
    intrinsics: torch.Tensor,
    pixels: torch.Tensor,
    mask: torch.Tensor | None = None,
    initial: torch.Tensor | None = None,
    iterations: int = 10,
    damping: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize reprojection error by Levenberg-Marquardt.

    This is the maximum-likelihood point under isotropic Gaussian pixel noise,
    and unlike the other two solvers it is genuinely iterative: the Jacobian
    depends on the point through the perspective divide.

    Starts from :func:`triangulate_linear` unless ``initial`` is given.
    """
    if initial is None:
        points, valid = triangulate_linear(motors, intrinsics, pixels, mask)
    else:
        points = initial
        valid = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

    views, num, _ = pixels.shape
    rotations = _mot.motor_to_matrix(motors)[:, :3, :3]  # (V, 3, 3)
    fx = intrinsics[:, 0, 0][:, None]
    fy = intrinsics[:, 1, 1][:, None]
    eye = torch.eye(3, dtype=points.dtype, device=points.device)

    for _ in range(iterations):
        residual, in_front = reprojection_residuals(points, motors, intrinsics, pixels)
        weights = in_front.to(points.dtype)
        if mask is not None:
            weights = weights * mask.to(points.dtype)

        cam = _cam.world_to_camera(
            motors[:, None, :].expand(views, num, 8), points.expand(views, num, 3)
        )
        x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
        inv_z = 1.0 / z.clamp_min(_EPS)
        zero = torch.zeros_like(x)
        # d(pixel)/d(camera point) for a pinhole, then chain through R.
        du = torch.stack([fx * inv_z, zero, -fx * x * inv_z * inv_z], dim=-1)
        dv = torch.stack([zero, fy * inv_z, -fy * y * inv_z * inv_z], dim=-1)
        jac = torch.stack([du, dv], dim=-2) @ rotations[:, None]  # (V, N, 2, 3)

        jac = jac * weights[..., None, None]
        residual = residual * weights[..., None]
        lhs = (jac.transpose(-2, -1) @ jac).sum(dim=0) + damping * eye
        rhs = (jac.transpose(-2, -1) @ residual.unsqueeze(-1)).squeeze(-1).sum(dim=0)
        points = points - torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)

    return points, valid
