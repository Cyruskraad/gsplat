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

"""Tests for gsplat.photogrammetry.mesh_extraction, on fully synthetic data
(no trained splats / GPU needed): `_tsdf_fuse` and `extract_mesh_poisson`
reconstruct an analytic unit sphere from ray-traced depth maps / a point
cloud respectively.
"""

import numpy as np
import pytest

pytest.importorskip(
    "open3d", reason="open3d is not installed (pip install gsplat[mesh])"
)

from gsplat.photogrammetry.mesh_extraction import _tsdf_fuse, extract_mesh_poisson


def _make_sphere_views(num_views=10, radius=1.0, cam_dist=3.0, width=200, height=150):
    """Ray-traced (color, depth) views of a unit sphere centered at the origin,
    from cameras placed on a larger sphere around it.
    """
    K = np.array(
        [[220.0, 0, width / 2], [0, 220.0, height / 2], [0, 0, 1.0]], dtype=np.float64
    )
    views = []
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        phi = np.pi / 2 + 0.3 * np.sin(3 * theta)
        cam_pos = cam_dist * np.array(
            [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        )
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_c2w = np.stack([right, -up, forward], axis=1)
        R_w2c = R_c2w.T
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = -R_w2c @ cam_pos

        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x1 = (xs - K[0, 2] + 0.5) / K[0, 0]
        y1 = (ys - K[1, 2] + 0.5) / K[1, 1]
        dirs_cam = np.stack([x1, y1, np.ones_like(x1)], axis=-1)  # (H, W, 3)
        dirs_cam_norm = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
        dirs_world = np.einsum("ij,hwj->hwi", R_c2w, dirs_cam_norm)

        oc = cam_pos
        b = np.einsum("hwi,i->hw", dirs_world, oc) * 2
        c = np.dot(oc, oc) - radius**2
        disc = b**2 - 4 * c
        valid = disc >= 0
        sqrt_disc = np.sqrt(np.clip(disc, 0, None))
        t0 = (-b - sqrt_disc) / 2
        hit = valid & (t0 > 0)

        ray_depth = np.where(hit, t0, 0.0)
        # ray-depth -> z-depth: divide by |unnormalized camera-space dir| / z,
        # i.e. multiply by cos(angle to principal axis).
        cos_angle = 1.0 / np.linalg.norm(dirs_cam, axis=-1)
        z_depth = (ray_depth * cos_angle).astype(np.float32)

        color = np.zeros((height, width, 3), dtype=np.uint8)
        color[hit] = np.array([200, 120, 60], dtype=np.uint8)

        views.append({"color": color, "depth": z_depth, "K": K, "extrinsic": extrinsic})
    return views


def test_tsdf_fuse_reconstructs_sphere():
    views = _make_sphere_views()
    mesh = _tsdf_fuse(views, voxel_size=0.02, sdf_trunc=0.08, depth_trunc=10.0)

    assert len(mesh.vertices) > 50
    assert len(mesh.triangles) > 50

    verts = np.asarray(mesh.vertices)
    radii = np.linalg.norm(verts, axis=1)
    mean_radius_err = np.mean(np.abs(radii - 1.0))
    assert (
        mean_radius_err < 0.1
    ), f"mesh doesn't look like a unit sphere: {mean_radius_err}"


def test_extract_mesh_poisson_reconstructs_sphere():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(2000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    normals = points.copy()  # outward normals on a unit sphere == the points
    colors = np.tile(np.array([0.5, 0.5, 0.5]), (points.shape[0], 1))

    mesh = extract_mesh_poisson(points, colors, normals=normals, depth=6)

    assert len(mesh.vertices) > 50
    verts = np.asarray(mesh.vertices)
    radii = np.linalg.norm(verts, axis=1)
    mean_radius_err = np.mean(np.abs(radii - 1.0))
    assert (
        mean_radius_err < 0.15
    ), f"mesh doesn't look like a unit sphere: {mean_radius_err}"
