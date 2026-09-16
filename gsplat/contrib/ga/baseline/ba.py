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
"""Bundle adjustment with quaternion + translation cameras -- the control arm.

This is the comparison this project has to carry to mean anything. Motors are
isomorphic to unit dual quaternions and PGA bivectors are exactly se(3), so the
geometric-algebra bundle adjuster cannot reach a *better* optimum than this one;
claiming otherwise without running both would be measuring implementation
quality rather than mathematics.

Two deliberate choices keep the comparison honest:

1. **The solver is shared, not reimplemented.** Both arms call
   :func:`gsplat.contrib.ga.sfm._lm.schur_lm`, so damping schedule, stopping
   rule and linear algebra are identical and the pose parameterization is the
   only variable.
2. **The pose math here is written from scratch** -- quaternion product,
   rotation, and Rodrigues exponential -- rather than delegating to
   :mod:`gsplat.contrib.ga.motor`. A control that shares the code under test
   cannot detect a bug in it.

Poses are ``(V, 7)`` as ``[qw, qx, qy, qz, tx, ty, tz]``, camera-from-world, and
are updated by a left se(3) increment ``[omega, u]``:
``R <- exp(skew(omega)) R``, ``t <- exp(skew(omega)) t + u``. Under that
increment a camera-frame point moves as ``p' = exp(skew(omega)) p + u``, giving
``dp'/domega = -skew(p)`` and ``dp'/du = I``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from gsplat.contrib.ga.sfm import _lm

__all__ = [
    "QuaternionBundleProblem",
    "quat_rotate",
    "quat_multiply",
    "rotation_from_axis_angle",
    "reprojection_residuals",
    "bundle_adjust",
    "poses_from_matrices",
    "poses_to_matrices",
]

_EPS = 1e-12


@dataclass
class QuaternionBundleProblem:
    """The same problem as :class:`gsplat.contrib.ga.sfm.ba.BundleProblem`,
    with poses stored as ``(V, 7)`` quaternion-translation instead of motors."""

    poses: torch.Tensor
    intrinsics: torch.Tensor
    points: torch.Tensor
    pixels: torch.Tensor
    camera_idx: torch.Tensor
    point_idx: torch.Tensor

    @property
    def num_cameras(self) -> int:
        return self.poses.shape[0]

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def num_observations(self) -> int:
        return self.pixels.shape[0]


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of ``wxyz`` quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by ``wxyz`` quaternions."""
    w, xyz = q[..., 0:1], q[..., 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """``wxyz`` quaternion -> ``(..., 3, 3)`` rotation matrix."""
    eye = torch.eye(3, dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 3, 3)
    return torch.stack(
        [quat_rotate(q, eye[..., :, i]) for i in range(3)], dim=-1
    )


def rotation_from_axis_angle(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential: axis-angle ``(..., 3)`` -> ``wxyz`` quaternion."""
    angle = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    half = angle / 2.0
    # sin(half)/angle, continued to 1/2 at angle = 0.
    small = angle < 1e-8
    scale = torch.where(small, torch.full_like(angle, 0.5), torch.sin(half) / angle.clamp_min(_EPS))
    return torch.cat([torch.cos(half), omega * scale], dim=-1)


def poses_from_matrices(matrices: torch.Tensor) -> torch.Tensor:
    """``(V, 4, 4)`` -> ``(V, 7)`` quaternion-translation poses."""
    rotation, translation = matrices[..., :3, :3], matrices[..., :3, 3]
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    w = torch.sqrt((1.0 + trace).clamp_min(_EPS)) / 2.0
    quat = torch.stack(
        [
            w,
            (rotation[..., 2, 1] - rotation[..., 1, 2]) / (4.0 * w),
            (rotation[..., 0, 2] - rotation[..., 2, 0]) / (4.0 * w),
            (rotation[..., 1, 0] - rotation[..., 0, 1]) / (4.0 * w),
        ],
        dim=-1,
    )
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cat([quat, translation], dim=-1)


def poses_to_matrices(poses: torch.Tensor) -> torch.Tensor:
    """``(V, 7)`` -> ``(V, 4, 4)``."""
    batch = poses.shape[:-1]
    out = torch.zeros(*batch, 4, 4, dtype=poses.dtype, device=poses.device)
    out[..., :3, :3] = quat_to_matrix(poses[..., :4])
    out[..., :3, 3] = poses[..., 4:]
    out[..., 3, 3] = 1.0
    return out


def _project(poses: torch.Tensor, intrinsics: torch.Tensor, points: torch.Tensor):
    cam = quat_rotate(poses[..., :4], points) + poses[..., 4:]
    depth = cam[..., 2]
    valid = depth > _EPS
    safe = torch.where(valid, depth, torch.ones_like(depth))
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    u = fx * cam[..., 0] / safe + cx
    v = fy * cam[..., 1] / safe + cy
    return torch.stack([u, v], dim=-1), cam, valid


def reprojection_residuals(problem: QuaternionBundleProblem) -> torch.Tensor:
    """Per-observation reprojection residuals ``(M, 2)`` in pixels."""
    pixels, _, _ = _project(
        problem.poses[problem.camera_idx],
        problem.intrinsics[problem.camera_idx],
        problem.points[problem.point_idx],
    )
    return pixels - problem.pixels


def _skew(v: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def _residual_and_jacobians(problem: QuaternionBundleProblem):
    poses = problem.poses[problem.camera_idx]
    intrinsics = problem.intrinsics[problem.camera_idx]
    points = problem.points[problem.point_idx]

    pixels, cam_points, _ = _project(poses, intrinsics, points)
    residual = pixels - problem.pixels

    x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]
    inv_z = 1.0 / z.clamp_min(_EPS)
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    zero = torch.zeros_like(x)
    d_pixel = torch.stack(
        [
            torch.stack([fx * inv_z, zero, -fx * x * inv_z * inv_z], dim=-1),
            torch.stack([zero, fy * inv_z, -fy * y * inv_z * inv_z], dim=-1),
        ],
        dim=-2,
    )

    eye = torch.eye(3, dtype=points.dtype, device=points.device).expand(
        problem.num_observations, 3, 3
    )
    d_cam_d_delta = torch.cat([-_skew(cam_points), eye], dim=-1)
    rotations = quat_to_matrix(problem.poses[:, :4])[problem.camera_idx]
    return residual, d_pixel @ d_cam_d_delta, d_pixel @ rotations


def bundle_adjust(
    problem: QuaternionBundleProblem,
    iterations: int = 30,
    fixed_cameras: tuple[int, ...] = (0,),
    damping: float = 1e-4,
    tolerance: float = 1e-12,
) -> tuple[QuaternionBundleProblem, dict]:
    """Refine cameras and points, using the same solver as the motor arm."""

    def apply_step(state, delta_cam, delta_pt):
        omega, u = delta_cam[:, :3], delta_cam[:, 3:]
        delta_q = rotation_from_axis_angle(omega)
        quat = quat_multiply(delta_q, state.poses[:, :4])
        # Unit norm is a *constraint* here, so it has to be re-imposed every
        # step -- the cost the bivector parameterization does not pay.
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(_EPS)
        translation = quat_rotate(delta_q, state.poses[:, 4:]) + u
        return replace(
            state,
            poses=torch.cat([quat, translation], dim=-1),
            points=state.points + delta_pt,
        )

    def cost(state):
        return reprojection_residuals(state).pow(2).sum()

    state, stats = _lm.schur_lm(
        problem,
        _residual_and_jacobians,
        apply_step,
        cost,
        num_cameras=problem.num_cameras,
        num_points=problem.num_points,
        camera_idx=problem.camera_idx,
        point_idx=problem.point_idx,
        iterations=iterations,
        fixed_cameras=fixed_cameras,
        damping=damping,
        tolerance=tolerance,
    )
    stats["rmse"] = float(
        (torch.as_tensor(stats["final_cost"]) / max(state.num_observations, 1)).sqrt()
    )
    return state, stats
