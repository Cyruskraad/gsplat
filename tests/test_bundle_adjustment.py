# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for gsplat.photogrammetry.bundle_adjustment, on a fully synthetic
scene (no COLMAP files / pycolmap needed): the pure-torch `_optimize` routine
is exercised directly.
"""

import numpy as np
import torch

from gsplat.photogrammetry.bundle_adjustment import _optimize, _so3_exp

# `_optimize` has no CUDA-only ops; run on CPU in CI/dev environments without
# a GPU and on CUDA when available, unlike the rest of gsplat's (CUDA-kernel)
# test suite.
device = "cuda" if torch.cuda.is_available() else "cpu"


def _make_synthetic_scene(num_cameras: int = 6, num_points: int = 80, seed: int = 0):
    """A handful of cameras on a circle looking at the origin, viewing random
    3D points. Returns GT (w2c, Ks, points) plus noiseless 2D observations.
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(num_points, 3)).astype(np.float32)

    angles = np.linspace(0, 2 * np.pi, num_cameras, endpoint=False)
    radius = 4.0
    w2c = np.zeros((num_cameras, 4, 4), dtype=np.float32)
    for i, theta in enumerate(angles):
        cam_pos = np.array(
            [radius * np.cos(theta), radius * np.sin(theta), 0.5], dtype=np.float32
        )
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_c2w = np.stack([right, -up, forward], axis=1)
        R_w2c = R_c2w.T
        w2c[i, :3, :3] = R_w2c
        w2c[i, :3, 3] = -R_w2c @ cam_pos
        w2c[i, 3, 3] = 1.0

    K = np.array([[300.0, 0, 150.0], [0, 300.0, 100.0], [0, 0, 1.0]], dtype=np.float32)
    Ks = np.stack([K] * num_cameras, axis=0)

    obs_image_idx, obs_point_idx, obs_xy = [], [], []
    for i in range(num_cameras):
        Xc = (w2c[i, :3, :3] @ points.T).T + w2c[i, :3, 3]
        valid = Xc[:, 2] > 0.1
        uvw = (K @ Xc.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        in_bounds = (
            (uv[:, 0] >= 0) & (uv[:, 0] < 300) & (uv[:, 1] >= 0) & (uv[:, 1] < 200)
        )
        for p in np.nonzero(valid & in_bounds)[0]:
            obs_image_idx.append(i)
            obs_point_idx.append(int(p))
            obs_xy.append(uv[p])

    return (
        w2c,
        Ks,
        points,
        np.array(obs_image_idx),
        np.array(obs_point_idx),
        np.array(obs_xy, dtype=np.float32),
    )


def test_optimize_noop_at_ground_truth():
    """Reprojecting with the exact GT poses/points should have ~0 error."""
    w2c, Ks, points, obs_image_idx, obs_point_idx, obs_xy = _make_synthetic_scene()
    assert len(obs_image_idx) > 50

    result = _optimize(
        w2c_init=torch.from_numpy(w2c).to(device),
        Ks=torch.from_numpy(Ks).to(device),
        points_init=torch.from_numpy(points).to(device),
        obs_image_idx=torch.from_numpy(obs_image_idx).long().to(device),
        obs_point_idx=torch.from_numpy(obs_point_idx).long().to(device),
        obs_xy=torch.from_numpy(obs_xy).to(device),
        anchor_image_idx=0,
        num_iters=0,
        refine_points=True,
    )
    assert result["mean_reprojection_error_before"] < 1e-3


def test_optimize_recovers_perturbed_poses_and_points():
    """Perturb poses (except a fixed anchor) and points, then check that
    bundle adjustment recovers both the reprojection error and the ground
    truth camera rotations.
    """
    w2c, Ks, points, obs_image_idx, obs_point_idx, obs_xy = _make_synthetic_scene()

    rng = np.random.default_rng(1)
    w2c_noisy = w2c.copy()
    for i in range(1, w2c.shape[0]):
        noise_r = rng.normal(scale=0.03, size=3).astype(np.float32)
        noise_t = rng.normal(scale=0.02, size=3).astype(np.float32)
        R_noise = _so3_exp(torch.from_numpy(noise_r)).numpy()
        w2c_noisy[i, :3, :3] = R_noise @ w2c[i, :3, :3]
        w2c_noisy[i, :3, 3] = w2c[i, :3, 3] + noise_t
    points_noisy = points + rng.normal(scale=0.02, size=points.shape).astype(
        np.float32
    )

    result = _optimize(
        w2c_init=torch.from_numpy(w2c_noisy).to(device),
        Ks=torch.from_numpy(Ks).to(device),
        points_init=torch.from_numpy(points_noisy).to(device),
        obs_image_idx=torch.from_numpy(obs_image_idx).long().to(device),
        obs_point_idx=torch.from_numpy(obs_point_idx).long().to(device),
        obs_xy=torch.from_numpy(obs_xy).to(device),
        anchor_image_idx=0,
        num_iters=800,
        lr=1e-2,
        huber_delta=1.0,
        anchor_reg=1e-5,
        refine_points=True,
    )

    # Reprojection error should drop by >90% and land well under a pixel.
    assert (
        result["mean_reprojection_error_after"]
        < 0.1 * result["mean_reprojection_error_before"]
    )
    assert result["mean_reprojection_error_after"] < 1.0

    # The anchor image's pose must be left exactly untouched (gauge fix).
    w2c_refined = result["w2c"].cpu().numpy()
    np.testing.assert_allclose(w2c_refined[0], w2c_noisy[0], atol=1e-6)

    # Recovered rotations should match ground truth closely.
    for i in range(1, w2c.shape[0]):
        R_err = w2c_refined[i, :3, :3] @ w2c[i, :3, :3].T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        assert angle < 0.02, f"camera {i} rotation error too large: {angle} rad"


def test_optimize_refine_points_false_keeps_points_fixed():
    w2c, Ks, points, obs_image_idx, obs_point_idx, obs_xy = _make_synthetic_scene()
    result = _optimize(
        w2c_init=torch.from_numpy(w2c).to(device),
        Ks=torch.from_numpy(Ks).to(device),
        points_init=torch.from_numpy(points).to(device),
        obs_image_idx=torch.from_numpy(obs_image_idx).long().to(device),
        obs_point_idx=torch.from_numpy(obs_point_idx).long().to(device),
        obs_xy=torch.from_numpy(obs_xy).to(device),
        anchor_image_idx=0,
        num_iters=10,
        refine_points=False,
    )
    np.testing.assert_allclose(result["points"].cpu().numpy(), points, atol=1e-6)
