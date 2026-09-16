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
"""Bundle adjustment with motor-parameterized cameras.

Poses are updated by a **left bivector increment**: ``M <- exp(delta) M``, with
``delta`` a 6-component PGA bivector. That parameterization is the practical
argument for motors in an optimizer -- it is minimal (6 numbers for 6 degrees of
freedom), unconstrained (no unit-norm to maintain, no renormalization between
steps), and free of the quaternion sign ambiguity, because the increment lives
in the tangent algebra rather than on a manifold embedded in a larger space.

The increment acts on a camera-frame point in closed form. Writing ``delta`` as
``[w, v]``, the derivative at ``delta = 0`` is

    d(p_cam)/dw = 2 * skew(p_cam)      d(p_cam)/dv = -2 * I

(the factors of two are the motor half-angle convention). Both are checked
against autograd in ``tests/ga/test_ba.py`` rather than trusted.

Solver: Levenberg-Marquardt on the normal equations, reduced by the Schur
complement. The point block is block-diagonal and inverts per-point; eliminating
it leaves a system in the cameras alone. That matters -- forming the full
Jacobian densely for even a modest problem (20 cameras, 2000 points) would be
several hundred megabytes.

Gauge: bundle adjustment is invariant to a global similarity, so the normal
equations are rank-deficient by 7 without a constraint. Cameras listed in
``fixed_cameras`` are held (removing 6), and the LM damping regularizes the
remaining scale direction. Compare reconstructions only after aligning them --
:func:`align_similarity` is provided for exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from gsplat.contrib.ga import camera as _cam
from gsplat.contrib.ga import primitives as _prim
from gsplat.contrib.ga import motor as _mot
from gsplat.contrib.ga.sfm import _lm

__all__ = [
    "BundleProblem",
    "LineBundleProblem",
    "canonical_line",
    "line_residuals",
    "bundle_adjust_lines",
    "reprojection_residuals",
    "bundle_adjust",
    "align_similarity",
]

_EPS = 1e-12


@dataclass
class BundleProblem:
    """A sparse bundle-adjustment problem.

    Observations are a flat list rather than a dense ``(views, points)`` grid:
    real tracks are sparse, and a grid would spend most of its memory on
    absent observations.

    Attributes:
        motors: camera-from-world motors ``(V, 8)``.
        intrinsics: pinhole ``K`` per camera ``(V, 3, 3)``.
        points: world points ``(P, 3)``.
        pixels: observed pixel coordinates ``(M, 2)``.
        camera_idx: ``(M,)`` index into ``motors`` per observation.
        point_idx: ``(M,)`` index into ``points`` per observation.
    """

    motors: torch.Tensor
    intrinsics: torch.Tensor
    points: torch.Tensor
    pixels: torch.Tensor
    camera_idx: torch.Tensor
    point_idx: torch.Tensor

    @property
    def num_cameras(self) -> int:
        return self.motors.shape[0]

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def num_observations(self) -> int:
        return self.pixels.shape[0]


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


def reprojection_residuals(problem: BundleProblem) -> torch.Tensor:
    """Per-observation reprojection residuals ``(M, 2)`` in pixels."""
    motors = problem.motors[problem.camera_idx]
    intrinsics = problem.intrinsics[problem.camera_idx]
    points = problem.points[problem.point_idx]
    projected, _ = _cam.project(motors, intrinsics, points)
    return projected - problem.pixels


def _residual_and_jacobians(
    problem: BundleProblem,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residuals ``(M, 2)`` with camera ``(M, 2, 6)`` and point ``(M, 2, 3)`` Jacobians."""
    motors = problem.motors[problem.camera_idx]
    intrinsics = problem.intrinsics[problem.camera_idx]
    points = problem.points[problem.point_idx]

    cam_points = _cam.world_to_camera(motors, points)
    pixels, _ = _cam.project(motors, intrinsics, points)
    residual = pixels - problem.pixels

    x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]
    inv_z = 1.0 / z.clamp_min(_EPS)
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    zero = torch.zeros_like(x)
    # d(pixel)/d(camera point) for a pinhole.
    d_pixel = torch.stack(
        [
            torch.stack([fx * inv_z, zero, -fx * x * inv_z * inv_z], dim=-1),
            torch.stack([zero, fy * inv_z, -fy * y * inv_z * inv_z], dim=-1),
        ],
        dim=-2,
    )  # (M, 2, 3)

    # d(camera point)/d(left bivector increment), verified against autograd.
    eye = torch.eye(3, dtype=points.dtype, device=points.device).expand(
        problem.num_observations, 3, 3
    )
    d_cam_d_delta = torch.cat([2.0 * _skew(cam_points), -2.0 * eye], dim=-1)  # (M, 3, 6)

    rotations = _mot.motor_to_matrix(problem.motors)[:, :3, :3][problem.camera_idx]
    return residual, d_pixel @ d_cam_d_delta, d_pixel @ rotations


def bundle_adjust(
    problem: BundleProblem,
    iterations: int = 30,
    fixed_cameras: tuple[int, ...] = (0,),
    damping: float = 1e-4,
    tolerance: float = 1e-12,
) -> tuple[BundleProblem, dict]:
    """Refine cameras and points by Levenberg-Marquardt.

    The solver itself lives in :mod:`gsplat.contrib.ga.sfm._lm` and is shared
    verbatim with the quaternion/se(3) control in
    :mod:`gsplat.contrib.ga.baseline.ba`, so the pose parameterization is the
    only difference between the two arms.

    Returns the refined problem and a stats dict including the cost history.
    """

    def apply_step(
        state: BundleProblem, delta_cam: torch.Tensor, delta_pt: torch.Tensor
    ) -> BundleProblem:
        # Left increment in the tangent algebra: M <- exp(delta) M. No
        # constraint to project back onto; normalize only to shed float drift.
        return replace(
            state,
            motors=_mot.motor_normalize(
                _mot.motor_compose(_mot.motor_exp(delta_cam), state.motors)
            ),
            points=state.points + delta_pt,
        )

    def cost(state: BundleProblem) -> torch.Tensor:
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


def align_similarity(
    source: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Umeyama similarity alignment of ``source`` onto ``target``.

    Bundle adjustment fixes geometry only up to a global similarity, so two
    correct reconstructions can differ by one. Comparing them without aligning
    first measures the gauge, not the estimate.

    Returns the aligned source and the RMSE after alignment.
    """
    src_mean, dst_mean = source.mean(0), target.mean(0)
    src_c, dst_c = source - src_mean, target - dst_mean
    covariance = dst_c.T @ src_c / source.shape[0]
    u, s, vh = torch.linalg.svd(covariance)
    d = torch.ones(3, dtype=source.dtype, device=source.device)
    if torch.det(u @ vh) < 0:
        d[-1] = -1.0
    rotation = u @ torch.diag(d) @ vh
    variance = src_c.pow(2).sum() / source.shape[0]
    scale = float((s * d).sum() / variance.clamp_min(_EPS))
    aligned = scale * (src_c @ rotation.T) + dst_mean
    return aligned, float((aligned - target).norm(dim=-1).pow(2).mean().sqrt())


#: The canonical line every reconstructed line is a motor away from: the z axis.
#: A PGA line has six coefficients but only four degrees of freedom, so it is
#: carried by a motor rather than parameterized directly -- the same motor
#: machinery the cameras use, with the same bivector increment. The two extra
#: parameters are the screw motions that slide a line along itself, which leave
#: it unchanged; they are null directions of the normal equations and are
#: absorbed by the solver's damping, exactly like the global gauge.
def canonical_line(dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    return _prim.line_from_point_direction(
        torch.zeros(3, dtype=dtype, device=device),
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device),
    )


@dataclass
class LineBundleProblem:
    """Bundle adjustment over 3D *line* features.

    Attributes:
        motors: camera-from-world motors ``(V, 8)``.
        intrinsics: pinhole ``K`` per camera ``(V, 3, 3)``.
        line_motors: ``(L, 8)``; line ``i`` is ``line_motors[i]`` applied to
            :func:`canonical_line`.
        image_lines: observed homogeneous 2D lines ``(M, 3)``.
        camera_idx, line_idx: ``(M,)`` indices per observation.
    """

    motors: torch.Tensor
    intrinsics: torch.Tensor
    line_motors: torch.Tensor
    image_lines: torch.Tensor
    camera_idx: torch.Tensor
    line_idx: torch.Tensor

    @property
    def num_cameras(self) -> int:
        return self.motors.shape[0]

    @property
    def num_lines(self) -> int:
        return self.line_motors.shape[0]

    @property
    def num_observations(self) -> int:
        return self.image_lines.shape[0]

    def world_lines(self) -> torch.Tensor:
        """The reconstructed lines ``(L, 6)`` in world coordinates."""
        base = canonical_line(self.line_motors.dtype, self.line_motors.device)
        return _mot.motor_apply_line(
            self.line_motors, base.expand(self.num_lines, 6)
        )


def _line_observation_residual(
    delta_cam: torch.Tensor,
    delta_line: torch.Tensor,
    camera_motor: torch.Tensor,
    line_motor: torch.Tensor,
    plane: torch.Tensor,
    base_line: torch.Tensor,
) -> torch.Tensor:
    """Residual of one line observation at increments ``(delta_cam, delta_line)``.

    Written as a function of the increments so ``torch.func.jacrev`` can
    differentiate it directly. That is the practical dividend of claim C1: the
    line residual reuses the point residual's wedge, and its Jacobian comes from
    autograd rather than from a second hand-derivation.
    """
    line = _mot.motor_apply_line(
        _mot.motor_compose(_mot.motor_exp(delta_line), line_motor), base_line
    )
    cam_line = _mot.motor_apply_line(
        _mot.motor_compose(_mot.motor_exp(delta_cam), camera_motor), line
    )
    return _prim.line_plane_residual(cam_line, plane)


def line_residuals(problem: LineBundleProblem) -> torch.Tensor:
    """Per-observation line residuals ``(M, 4)``; the norm of each is an offset."""
    planes = _cam.image_line_plane(
        problem.intrinsics[problem.camera_idx], problem.image_lines
    )
    world = problem.world_lines()[problem.line_idx]
    cam_lines = _mot.motor_apply_line(problem.motors[problem.camera_idx], world)
    return _prim.line_plane_residual(cam_lines, planes)


def _line_residual_and_jacobians(problem: LineBundleProblem):
    from torch.func import jacrev, vmap

    num = problem.num_observations
    dtype, device = problem.image_lines.dtype, problem.image_lines.device
    base = canonical_line(dtype, device)
    planes = _cam.image_line_plane(
        problem.intrinsics[problem.camera_idx], problem.image_lines
    )
    camera_motors = problem.motors[problem.camera_idx]
    line_motors = problem.line_motors[problem.line_idx]
    zeros = torch.zeros(num, 6, dtype=dtype, device=device)

    residual = vmap(_line_observation_residual, in_dims=(0, 0, 0, 0, 0, None))(
        zeros, zeros, camera_motors, line_motors, planes, base
    )
    jac_cam = vmap(jacrev(_line_observation_residual, argnums=0), in_dims=(0, 0, 0, 0, 0, None))(
        zeros, zeros, camera_motors, line_motors, planes, base
    )
    jac_line = vmap(jacrev(_line_observation_residual, argnums=1), in_dims=(0, 0, 0, 0, 0, None))(
        zeros, zeros, camera_motors, line_motors, planes, base
    )
    return residual, jac_cam, jac_line


def bundle_adjust_lines(
    problem: LineBundleProblem,
    iterations: int = 30,
    fixed_cameras: tuple[int, ...] = (0,),
    damping: float = 1e-4,
    tolerance: float = 1e-12,
) -> tuple[LineBundleProblem, dict]:
    """Refine cameras and 3D lines, using the same solver as the point arm.

    Only the residual and the structure-block width differ from
    :func:`bundle_adjust` -- there is no separate line solver, no Pluecker
    bookkeeping and no second Jacobian derivation. That is the concrete form of
    the claim that motivates doing this in geometric algebra.
    """

    def apply_step(state: LineBundleProblem, delta_cam, delta_line) -> LineBundleProblem:
        return replace(
            state,
            motors=_mot.motor_normalize(
                _mot.motor_compose(_mot.motor_exp(delta_cam), state.motors)
            ),
            line_motors=_mot.motor_normalize(
                _mot.motor_compose(_mot.motor_exp(delta_line), state.line_motors)
            ),
        )

    def cost(state: LineBundleProblem) -> torch.Tensor:
        return line_residuals(state).pow(2).sum()

    state, stats = _lm.schur_lm(
        problem,
        _line_residual_and_jacobians,
        apply_step,
        cost,
        num_cameras=problem.num_cameras,
        num_points=problem.num_lines,
        camera_idx=problem.camera_idx,
        point_idx=problem.line_idx,
        structure_dim=6,
        iterations=iterations,
        fixed_cameras=fixed_cameras,
        damping=damping,
        tolerance=tolerance,
    )
    stats["rmse"] = float(
        (torch.as_tensor(stats["final_cost"]) / max(state.num_observations, 1)).sqrt()
    )
    return state, stats
