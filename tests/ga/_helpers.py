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
"""Shared helpers for the geometric-algebra tests.

These tests are CPU-only by design: the GA layer is plain PyTorch, so it must be
verifiable without a GPU (unlike ``gsplat.geometry``, whose operators are
CUDA-only and whose suites skip without a device).
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["expm", "se3_matrix", "skew", "random_bivectors", "MAX_THETA"]

#: Motors double-cover SE(3); ``motor_log`` returns the ``scalar >= 0`` branch,
#: which caps the recoverable rotation magnitude at ``pi/2`` in bivector terms
#: (a half-turn of the underlying rotation).
MAX_THETA = np.pi / 2


def skew(w: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix of a 3-vector."""
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=float
    )


def expm(a: np.ndarray, terms: int = 60) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring plus a Taylor series.

    Written out rather than taken from SciPy so the oracle stays independent of
    the optional ``scipy`` extra, and so the test does not lean on a library
    that could share a bug with the code under test.
    """
    scale = max(0, int(np.ceil(np.log2(max(np.abs(a).max(), 1e-12)))) + 4)
    small = a / 2**scale
    out = np.eye(a.shape[0])
    term = np.eye(a.shape[0])
    for k in range(1, terms):
        term = term @ small / k
        out = out + term
    for _ in range(scale):
        out = out @ out
    return out


def se3_matrix(biv: torch.Tensor) -> np.ndarray:
    """The 4x4 rigid transform that a PGA bivector's motor should reproduce.

    The motor convention carries the quaternion half-angle factor, so the
    bivector ``[w, v]`` corresponds to the ``se(3)`` twist ``(-2w, -2v)``.
    """
    b = biv.detach().cpu().numpy().astype(float)
    a = np.zeros((4, 4))
    a[:3, :3] = skew(-2.0 * b[:3])
    a[:3, 3] = -2.0 * b[3:]
    return expm(a)


def random_bivectors(n: int, generator: torch.Generator, trans_scale: float = 2.0):
    """Random bivectors inside the principal branch of ``motor_log``."""
    w = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    norms = w.norm(dim=-1, keepdim=True)
    capped = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (MAX_THETA * 0.999)
    w = w / norms * capped
    v = torch.randn(n, 3, generator=generator, dtype=torch.float64) * trans_scale
    return torch.cat([w, v], dim=-1)


def synthetic_scene(
    views: int = 6,
    points: int = 200,
    seed: int = 0,
    jitter: float = 0.25,
    distance: float = 8.0,
):
    """A small calibrated multi-view scene with known ground truth.

    Cameras are jittered around a common stand-off from the origin so every
    point projects in front of every camera; that keeps visibility out of the
    tests that are not about visibility.

    Returns ``(motors, intrinsics, points, pixels)`` with motors camera-from-world.
    """
    import torch as _torch

    from gsplat.contrib.ga import camera as _camera
    from gsplat.contrib.ga import motor as _motor

    gen = _torch.Generator().manual_seed(seed)
    intrinsics = _torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=_torch.float64
    ).expand(views, 3, 3).contiguous()

    stand_off = _torch.zeros(views, 6, dtype=_torch.float64)
    stand_off[:, 5] = -distance / 2.0  # bivector v = -t/2
    motors = _motor.motor_compose(
        _motor.motor_exp(stand_off),
        _motor.motor_exp(_torch.randn(views, 6, generator=gen, dtype=_torch.float64) * jitter),
    )
    world = _torch.randn(points, 3, generator=gen, dtype=_torch.float64) * 0.8
    pixels, _ = _camera.project(
        motors[:, None, :].expand(views, points, 8),
        intrinsics[:, None, :, :].expand(views, points, 3, 3),
        world.expand(views, points, 3),
    )
    return motors, intrinsics, world, pixels


def bundle_problem(
    views: int = 8,
    points: int = 200,
    seed: int = 0,
    pose_noise: float = 0.02,
    point_noise: float = 0.05,
    pixel_noise: float = 0.0,
):
    """A bundle-adjustment problem perturbed away from a known ground truth.

    Returns ``(ga_problem, quaternion_problem, truth)`` where both problems
    describe *the same* starting geometry in the two parameterizations, so the
    arms can be compared without the initialization confounding the result.
    ``truth`` is ``(motors, intrinsics, points)``.
    """
    import torch as _torch

    from gsplat.contrib.ga import motor as _motor
    from gsplat.contrib.ga.baseline import ba as _qt_ba
    from gsplat.contrib.ga.sfm import ba as _ga_ba

    motors, intrinsics, world, pixels = synthetic_scene(
        views=views, points=points, seed=seed
    )
    gen = _torch.Generator().manual_seed(seed + 1000)
    camera_idx = _torch.arange(views).repeat_interleave(points)
    point_idx = _torch.arange(points).repeat(views)
    observations = pixels.reshape(-1, 2)
    if pixel_noise:
        observations = observations + _torch.randn(
            observations.shape, generator=gen, dtype=_torch.float64
        ) * pixel_noise

    delta = _torch.randn(views, 6, generator=gen, dtype=_torch.float64) * pose_noise
    delta[0] = 0.0  # camera 0 is the gauge anchor; start it at the truth
    start_motors = _motor.motor_compose(_motor.motor_exp(delta), motors)
    start_points = world + _torch.randn(
        points, 3, generator=gen, dtype=_torch.float64
    ) * point_noise

    ga_problem = _ga_ba.BundleProblem(
        start_motors, intrinsics, start_points.clone(), observations, camera_idx, point_idx
    )
    qt_problem = _qt_ba.QuaternionBundleProblem(
        _qt_ba.poses_from_matrices(_motor.motor_to_matrix(start_motors)),
        intrinsics,
        start_points.clone(),
        observations,
        camera_idx,
        point_idx,
    )
    return ga_problem, qt_problem, (motors, intrinsics, world)


def line_scene(views: int = 6, lines: int = 40, seed: int = 0, spread: float = 0.5):
    """A multi-view scene of 3D *line* features with known ground truth.

    Returns ``(problem, true_motors, true_world_lines)`` where ``problem`` holds
    the exact cameras and lines, so it can be perturbed by the caller.
    """
    import torch as _torch

    from gsplat.contrib.ga import camera as _camera
    from gsplat.contrib.ga import motor as _motor
    from gsplat.contrib.ga.sfm import ba as _ba

    gen = _torch.Generator().manual_seed(seed)
    intrinsics = _torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=_torch.float64
    ).expand(views, 3, 3).contiguous()
    stand_off = _torch.zeros(views, 6, dtype=_torch.float64)
    stand_off[:, 5] = -4.0
    motors = _motor.motor_compose(
        _motor.motor_exp(stand_off),
        _motor.motor_exp(_torch.randn(views, 6, generator=gen, dtype=_torch.float64) * 0.2),
    )
    line_motors = _motor.motor_exp(
        _torch.randn(lines, 6, generator=gen, dtype=_torch.float64) * spread
    )
    base = _ba.canonical_line()
    world_lines = _motor.motor_apply_line(line_motors, base.expand(lines, 6))

    camera_idx = _torch.arange(views).repeat_interleave(lines)
    line_idx = _torch.arange(lines).repeat(views)
    image_lines = _camera.project_line(
        motors[camera_idx], intrinsics[camera_idx], world_lines[line_idx]
    )
    problem = _ba.LineBundleProblem(
        motors, intrinsics, line_motors, image_lines, camera_idx, line_idx
    )
    return problem, motors, world_lines


def recover_similarity(source, target):
    """Umeyama similarity ``(rotation, scale, src_mean, dst_mean)`` mapping source onto target."""
    import torch as _torch

    src_mean, dst_mean = source.mean(0), target.mean(0)
    src_c, dst_c = source - src_mean, target - dst_mean
    covariance = dst_c.T @ src_c / source.shape[0]
    u, s, vh = _torch.linalg.svd(covariance)
    d = _torch.ones(3, dtype=source.dtype)
    if _torch.det(u @ vh) < 0:
        d[-1] = -1.0
    rotation = u @ _torch.diag(d) @ vh
    scale = float((s * d).sum() / (src_c.pow(2).sum() / source.shape[0]))
    return rotation, scale, src_mean, dst_mean


def two_view_scene(
    points: int = 300,
    seed: int = 0,
    pixel_noise: float = 0.0,
    outlier_fraction: float = 0.0,
    image_size: tuple[float, float] = (640.0, 480.0),
):
    """A calibrated two-view pair with known relative pose.

    Camera A sits at the identity, so world coordinates *are* camera A's frame
    and the returned ``true_relative`` motor carries frame B back to frame A.

    Returns ``(points_a, points_b, intrinsics, true_relative, inlier_mask)``.
    """
    import torch as _torch

    from gsplat.contrib.ga import camera as _camera
    from gsplat.contrib.ga import motor as _motor

    gen = _torch.Generator().manual_seed(seed)
    intrinsics = _torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=_torch.float64
    )
    true_relative = _motor.motor_exp(
        _torch.tensor([0.06, -0.09, 0.03, 0.35, -0.15, 0.05], dtype=_torch.float64)
    )
    pose_b = _motor.motor_inverse(true_relative)

    world = _torch.randn(points, 3, generator=gen, dtype=_torch.float64)
    world[:, 2] += 6.0
    identity = _motor.motor_identity(dtype=_torch.float64)
    points_a, _ = _camera.project(
        identity.expand(points, 8), intrinsics.expand(points, 3, 3), world
    )
    points_b, _ = _camera.project(
        pose_b.expand(points, 8), intrinsics.expand(points, 3, 3), world
    )

    if pixel_noise:
        points_a = points_a + _torch.randn(
            points_a.shape, generator=gen, dtype=_torch.float64
        ) * pixel_noise
        points_b = points_b + _torch.randn(
            points_b.shape, generator=gen, dtype=_torch.float64
        ) * pixel_noise

    inliers = _torch.ones(points, dtype=_torch.bool)
    count = int(points * outlier_fraction)
    if count:
        index = _torch.randperm(points, generator=gen)[:count]
        points_b[index] = _torch.rand(
            count, 2, generator=gen, dtype=_torch.float64
        ) * _torch.tensor(image_size, dtype=_torch.float64)
        inliers[index] = False

    return points_a, points_b, intrinsics, true_relative, inliers


def pose_errors(estimate, reference):
    """``(rotation_deg, translation_direction_deg)`` between two relative motors.

    Two-view translation is only defined up to scale *and* sign, so the
    direction error folds the sign away.
    """
    import torch as _torch

    from gsplat.contrib.ga import motor as _motor

    est = _motor.motor_to_matrix(estimate)
    ref = _motor.motor_to_matrix(reference)
    cos_angle = ((_torch.trace(est[:3, :3].T @ ref[:3, :3]) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation = float(_torch.arccos(cos_angle) * 180.0 / _torch.pi)

    cosine = _torch.nn.functional.cosine_similarity(
        est[:3, 3].unsqueeze(0), ref[:3, 3].unsqueeze(0)
    ).abs().clamp(max=1.0)
    direction = float(_torch.arccos(cosine) * 180.0 / _torch.pi)
    return rotation, direction
