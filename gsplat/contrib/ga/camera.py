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
"""Pinhole cameras with motor extrinsics.

Extrinsics are motors; intrinsics are not. A pinhole projection is projective,
not a versor, so it cannot be written as a sandwich product -- pretending
otherwise is the usual way a "geometric algebra camera" quietly stops being
geometric algebra. Keeping the split explicit means the motor half stays exact
and the projective half stays honest.

Motors here are **camera-from-world** ("w2c"), matching the ``viewmats``
convention used across gsplat and COLMAP.

The payoff of the algebra shows up in :func:`pixel_ray`: a pixel back-projects
to a *line*, a first-class object, so triangulation becomes "find the point
closest to incident on these lines" with no Pluecker bookkeeping.
"""

from __future__ import annotations

import torch

from gsplat.contrib.ga import motor as _mot
from gsplat.contrib.ga import primitives as _prim

__all__ = [
    "camera_center",
    "world_to_camera",
    "camera_to_world",
    "project",
    "pixel_direction",
    "pixel_ray",
    "pixel_ray_planes",
]

_EPS = 1e-12


def _apply(motor: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Broadcast a motor over a batch of points."""
    if motor.dim() == 1 and points.dim() > 1:
        motor = motor.expand(*points.shape[:-1], 8)
    return _mot.motor_apply_point(motor, points)


def camera_center(motor: torch.Tensor) -> torch.Tensor:
    """World-space camera centre of a camera-from-world motor."""
    origin = torch.zeros(*motor.shape[:-1], 3, dtype=motor.dtype, device=motor.device)
    return _mot.motor_apply_point(_mot.motor_inverse(motor), origin)


def world_to_camera(motor: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform world points ``(..., 3)`` into the camera frame."""
    return _apply(motor, points)


def camera_to_world(motor: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform camera-frame points ``(..., 3)`` into the world frame."""
    return _apply(_mot.motor_inverse(motor), points)


def project(
    motor: torch.Tensor, intrinsics: torch.Tensor, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points to pixels.

    Args:
        motor: camera-from-world motor ``(..., 8)``.
        intrinsics: pinhole ``K`` of shape ``(..., 3, 3)``.
        points: world points ``(..., 3)``.

    Returns:
        ``(pixels, valid)`` where ``pixels`` is ``(..., 2)`` and ``valid`` marks
        the points in front of the camera. Points at or behind the pinhole have
        no projection; the depth is clamped so the returned value stays finite
        and differentiable, and callers are expected to mask with ``valid``
        rather than trust those entries.
    """
    cam = world_to_camera(motor, points)
    depth = cam[..., 2]
    valid = depth > _EPS
    safe = torch.where(valid, depth, torch.ones_like(depth))
    normalized = cam[..., :2] / safe.unsqueeze(-1)
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    skew = intrinsics[..., 0, 1]
    u = fx * normalized[..., 0] + skew * normalized[..., 1] + cx
    v = fy * normalized[..., 1] + cy
    return torch.stack([u, v], dim=-1), valid


def pixel_direction(intrinsics: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """Camera-frame direction ``(..., 3)`` for pixels ``(..., 2)`` (unnormalized, z=1)."""
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    skew = intrinsics[..., 0, 1]
    y = (pixels[..., 1] - cy) / fy
    x = (pixels[..., 0] - cx - skew * y) / fx
    return torch.stack([x, y, torch.ones_like(x)], dim=-1)


def pixel_ray(
    motor: torch.Tensor, intrinsics: torch.Tensor, pixels: torch.Tensor
) -> torch.Tensor:
    """Back-project pixels to world-space rays as lines ``(..., 6)``.

    The ray is built in the camera frame -- the line joining the pinhole to the
    point at ``z = 1`` along the pixel direction -- and then carried to the
    world by the motor. Transforming the *line* directly is the same sandwich
    that transforms points and planes.
    """
    direction = pixel_direction(intrinsics, pixels)
    origin = torch.zeros_like(direction)
    ray_cam = _prim.line_from_point_direction(origin, direction)
    inverse = _mot.motor_inverse(motor)
    if inverse.dim() == 1 and ray_cam.dim() > 1:
        inverse = inverse.expand(*ray_cam.shape[:-1], 8)
    return _prim.normalize_line(_mot.motor_apply_line(inverse, ray_cam))


def pixel_ray_planes(
    motor: torch.Tensor, intrinsics: torch.Tensor, pixels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two world planes whose meet is the pixel's ray.

    Each plane contributes one *linear* constraint on a 3D point, so stacking
    them over views gives linear triangulation -- the DLT system, arrived at
    through incidence rather than through a hand-built measurement matrix.
    :func:`gsplat.contrib.ga.primitives.point_plane_distance` turns each into a
    metric residual.
    """
    direction = pixel_direction(intrinsics, pixels)
    x, y = direction[..., 0], direction[..., 1]
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    # In the camera frame both planes pass through the pinhole (d = 0):
    #   x_c - x * z_c = 0   and   y_c - y * z_c = 0
    plane_u = torch.stack([one, zero, -x, zero], dim=-1)
    plane_v = torch.stack([zero, one, -y, zero], dim=-1)

    inverse = _mot.motor_inverse(motor)
    if inverse.dim() == 1 and plane_u.dim() > 1:
        inverse = inverse.expand(*plane_u.shape[:-1], 8)
    return (
        _prim.normalize_plane(_mot.motor_apply_plane(inverse, plane_u)),
        _prim.normalize_plane(_mot.motor_apply_plane(inverse, plane_v)),
    )
