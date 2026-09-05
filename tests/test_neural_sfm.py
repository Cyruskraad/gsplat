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

"""Tests for gsplat.photogrammetry.neural_sfm, on synthetic multi-view point
maps (no real DUSt3R/MASt3R/VGGT tool needed -- this module is a tool-agnostic
adapter, see its docstring). `merge_point_maps_to_tracks` is pure numpy/
scikit-learn and tested directly; `write_colmap_reconstruction` needs
pycolmap, and is tested end-to-end including feeding its output into
`bundle_adjustment.refine_reconstruction`, proving the two compose as
intended.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed")

from gsplat.photogrammetry.neural_sfm import (
    merge_point_maps_to_tracks,
    write_colmap_reconstruction,
)


def _make_synthetic_cameras(num_cameras=6, radius=4.0):
    angles = np.linspace(0, 2 * np.pi, num_cameras, endpoint=False)
    camtoworlds, w2c_list = [], []
    for theta in angles:
        cam_pos = np.array([radius * np.cos(theta), radius * np.sin(theta), 0.5])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_c2w = np.stack([right, -up, forward], axis=1)
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = cam_pos
        camtoworlds.append(c2w)
        w2c_list.append((R_c2w.T, -R_c2w.T @ cam_pos))
    return np.stack(camtoworlds, axis=0), w2c_list


def _project(points_world, R_w2c, t_w2c, K, width, height):
    Xc = (R_w2c @ points_world.T).T + t_w2c
    uvw = (K @ Xc.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    valid = Xc[:, 2] > 0.1
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv, valid & in_bounds


def test_merge_point_maps_to_tracks_merges_shared_points():
    """Points independently 're-observed' (with small noise) by several
    synthetic cameras should merge into one multi-image track each; a point
    seen by only one camera should stay a length-1 track.
    """
    rng = np.random.default_rng(0)
    num_cameras = 5
    shared_points = rng.uniform(-1, 1, size=(20, 3))

    points_per_image, pixel_xy_per_image = [], []
    for i in range(num_cameras):
        noisy = shared_points + rng.normal(scale=0.001, size=shared_points.shape)
        pixel_xy_per_image.append(rng.uniform(0, 100, size=(20, 2)))
        points_per_image.append(noisy)

    # One extra point only seen by camera 0 (should stay track length 1).
    points_per_image[0] = np.concatenate(
        [points_per_image[0], [[5.0, 5.0, 5.0]]], axis=0
    )
    pixel_xy_per_image[0] = np.concatenate(
        [pixel_xy_per_image[0], [[10.0, 10.0]]], axis=0
    )

    result = merge_point_maps_to_tracks(
        points_per_image, pixel_xy_per_image, merge_radius=0.05
    )
    track_lengths = sorted(len(t) for t in result["tracks"])
    assert track_lengths.count(num_cameras) == 20
    assert track_lengths.count(1) == 1

    # min_track_length filters out the singleton.
    result_filtered = merge_point_maps_to_tracks(
        points_per_image, pixel_xy_per_image, merge_radius=0.05, min_track_length=2
    )
    assert len(result_filtered["tracks"]) == 20
    assert all(len(t) == num_cameras for t in result_filtered["tracks"])


def test_merge_point_maps_to_tracks_does_not_chain():
    """Single-linkage clustering (connected-components on a radius graph) can
    chain points transitively into one giant cluster even when its endpoints
    are much farther apart than merge_radius -- e.g. a dense line of points
    each merge_radius/2 apart. merge_point_maps_to_tracks must not do this:
    every merged track's points should be mutually within merge_radius.
    """
    merge_radius = 0.1
    # A chain of 20 points spaced merge_radius/2 apart along the x-axis:
    # adjacent points are well within merge_radius of each other, but the
    # first and last points are ~1.9 apart -- 19x merge_radius.
    chain = np.stack(
        [np.arange(20) * (merge_radius / 2), np.zeros(20), np.zeros(20)], axis=1
    )
    points_per_image = [chain[i : i + 1] for i in range(20)]
    pixel_xy_per_image = [np.array([[float(i), 0.0]]) for i in range(20)]

    result = merge_point_maps_to_tracks(
        points_per_image, pixel_xy_per_image, merge_radius=merge_radius
    )

    # Reconstruct which original chain points landed in each track (via the
    # image index each observation came from -- image i only ever contained
    # chain point i) and assert every track's span stays within merge_radius.
    for track in result["tracks"]:
        member_positions = chain[[obs[0] for obs in track]]
        pairwise_span = np.linalg.norm(
            member_positions[:, None, :] - member_positions[None, :, :], axis=-1
        )
        assert pairwise_span.max() <= merge_radius + 1e-9, (
            f"track spans {pairwise_span.max():.3f}, exceeding merge_radius "
            f"{merge_radius} -- points were chained rather than merged"
        )
    # And no information was lost: every chain point appears in exactly one track.
    assert sum(len(t) for t in result["tracks"]) == 20


def test_write_colmap_reconstruction_round_trip(tmp_path):
    pycolmap = pytest.importorskip("pycolmap", reason="pycolmap not installed")

    width, height = 64, 48
    K = np.array([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1.0]])
    num_cameras = 6
    camtoworlds, w2c_list = _make_synthetic_cameras(num_cameras)

    rng = np.random.default_rng(1)
    points_world = rng.uniform(-1, 1, size=(25, 3))

    points_per_image, pixel_xy_per_image = [], []
    for R_w2c, t_w2c in w2c_list:
        uv, valid = _project(points_world, R_w2c, t_w2c, K, width, height)
        sel = np.nonzero(valid)[0]
        points_per_image.append(points_world[sel])
        pixel_xy_per_image.append(uv[sel])

    merged = merge_point_maps_to_tracks(
        points_per_image, pixel_xy_per_image, merge_radius=1e-4, min_track_length=2
    )
    assert merged["points_xyz"].shape[0] > 0

    output_dir = str(tmp_path / "neural_sfm_colmap")
    image_names = [f"img{i:03d}.png" for i in range(num_cameras)]
    write_colmap_reconstruction(
        image_names=image_names,
        camtoworlds=camtoworlds,
        Ks=K,
        image_sizes=(width, height),
        points_xyz=merged["points_xyz"],
        tracks=merged["tracks"],
        output_dir=output_dir,
    )

    recon = pycolmap.Reconstruction(output_dir)
    assert recon.num_images() == num_cameras
    assert recon.num_points3D() == merged["points_xyz"].shape[0]
    for image_id, image in recon.images.items():
        assert image.name in image_names
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        assert cam_from_world.matrix().shape == (3, 4)


def test_neural_sfm_output_composes_with_bundle_adjustment(tmp_path):
    """The whole point of write_colmap_reconstruction: its output should be
    directly usable by bundle_adjustment.refine_reconstruction, and BA should
    actually improve poses that came in noisy (as feed-forward neural-SfM
    poses typically are).
    """
    pytest.importorskip("pycolmap", reason="pycolmap not installed")
    from gsplat.photogrammetry.bundle_adjustment import refine_reconstruction

    width, height = 64, 48
    K = np.array([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1.0]])
    num_cameras = 6
    camtoworlds, w2c_list = _make_synthetic_cameras(num_cameras)

    rng = np.random.default_rng(2)
    points_world = rng.uniform(-1, 1, size=(40, 3))

    points_per_image, pixel_xy_per_image = [], []
    for R_w2c, t_w2c in w2c_list:
        uv, valid = _project(points_world, R_w2c, t_w2c, K, width, height)
        sel = np.nonzero(valid)[0]
        points_per_image.append(points_world[sel])
        pixel_xy_per_image.append(uv[sel])

    merged = merge_point_maps_to_tracks(
        points_per_image, pixel_xy_per_image, merge_radius=1e-4, min_track_length=2
    )

    # Perturb the "neural-SfM" camera-to-world poses before writing, as a
    # feed-forward tool's poses would be approximate rather than exact.
    noisy_camtoworlds = camtoworlds.copy()
    for i in range(1, num_cameras):
        noisy_camtoworlds[i, :3, 3] += rng.normal(scale=0.05, size=3)

    output_dir = str(tmp_path / "neural_sfm_colmap")
    image_names = [f"img{i:03d}.png" for i in range(num_cameras)]
    write_colmap_reconstruction(
        image_names=image_names,
        camtoworlds=noisy_camtoworlds,
        Ks=K,
        image_sizes=(width, height),
        points_xyz=merged["points_xyz"],
        tracks=merged["tracks"],
        output_dir=output_dir,
    )

    refined_dir = str(tmp_path / "refined")
    stats = refine_reconstruction(
        colmap_dir=output_dir,
        output_dir=refined_dir,
        num_iters=500,
        lr=1e-2,
        device="cpu",
    )
    assert (
        stats["mean_reprojection_error_after"] < stats["mean_reprojection_error_before"]
    )
