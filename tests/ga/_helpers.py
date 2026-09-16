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
