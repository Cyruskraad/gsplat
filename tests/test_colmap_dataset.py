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

"""Tests for examples.datasets.colmap's Parser/Dataset, on a small synthetic
COLMAP reconstruction built directly with pycolmap (no real capture/CLI
needed): covers the `colmap_dir` and `dense_points_path` Parser overrides and
the `mono_depth_dir` Dataset addition, none of which had test coverage before
(the CUDA-dependent example trainers that exercise them can't run in most CI
environments; this test needs only CPU + pycolmap/opencv/piexif/scikit-learn).
"""

import os
import sys

import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../examples"))

# Skip at collection time (rather than ImportError) on environments that
# don't install examples/requirements.txt's full dependency set -- e.g.
# core_tests.yml only installs the `examples` extra (Pillow/tqdm/tyro/
# imageio), not pycolmap/opencv/piexif/scikit-learn.
pycolmap = pytest.importorskip(
    "pycolmap",
    reason="pycolmap not installed (pip install -r examples/requirements.txt)",
)
pytest.importorskip("cv2", reason="opencv not installed")
imageio = pytest.importorskip("imageio.v2", reason="imageio not installed")
pytest.importorskip("piexif", reason="piexif not installed")

from datasets.colmap import Dataset, Parser  # noqa: E402

WIDTH, HEIGHT = 64, 48
FX = FY = 50.0
CX, CY = WIDTH / 2, HEIGHT / 2


def _project(points_world, R_w2c, t_w2c):
    Xc = (R_w2c @ points_world.T).T + t_w2c
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
    uvw = (K @ Xc.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    valid = Xc[:, 2] > 0.1
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < WIDTH) & (uv[:, 1] >= 0) & (uv[:, 1] < HEIGHT)
    )
    return uv, valid & in_bounds


def _build_synthetic_reconstruction(output_dir, num_cameras=4, num_points=30, seed=0):
    """A small COLMAP model: `num_cameras` cameras on a circle looking at the
    origin, viewing `num_points` random 3D points, written to `output_dir`.
    """
    rng = np.random.default_rng(seed)
    recon = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, FX, WIDTH, HEIGHT
    )
    camera.params = np.array([FX, FY, CX, CY], dtype=np.float64)
    recon.add_camera_with_trivial_rig(camera)

    points_world = rng.uniform(-1, 1, size=(num_points, 3))
    angles = np.linspace(0, 2 * np.pi, num_cameras, endpoint=False)
    radius = 4.0

    image_ids, per_image_sel, per_image_uv = [], {}, {}
    for i, theta in enumerate(angles):
        cam_pos = np.array([radius * np.cos(theta), radius * np.sin(theta), 0.5])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_c2w = np.stack([right, -up, forward], axis=1)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ cam_pos

        uv, sel_mask = _project(points_world, R_w2c, t_w2c)
        sel = np.nonzero(sel_mask)[0]
        per_image_sel[i] = sel
        per_image_uv[i] = uv

        img = pycolmap.Image()
        img.image_id = i + 1
        img.camera_id = 1
        img.name = f"img{i:03d}.png"
        img.points2D = [pycolmap.Point2D(uv[p]) for p in sel]

        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R_w2c), t_w2c)
        recon.add_image_with_trivial_frame(img, cam_from_world)
        image_ids.append(img.image_id)

    for p_idx in range(num_points):
        track = pycolmap.Track()
        for i in range(num_cameras):
            pos = np.nonzero(per_image_sel[i] == p_idx)[0]
            if len(pos) == 0:
                continue
            track.add_element(pycolmap.TrackElement(image_ids[i], int(pos[0])))
        if track.length() == 0:
            continue
        color = np.array([128, 128, 128], dtype=np.uint8)
        recon.add_point3D(points_world[p_idx], track, color)

    os.makedirs(output_dir, exist_ok=True)
    recon.write(output_dir)
    return recon


@pytest.fixture
def synthetic_dataset(tmp_path):
    """A full <data_dir> (sparse/0/ + images/) for a small synthetic scene."""
    data_dir = str(tmp_path)
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    num_cameras = 4
    _build_synthetic_reconstruction(sparse_dir, num_cameras=num_cameras)

    rng = np.random.default_rng(1)
    for i in range(num_cameras):
        img = rng.uniform(0, 255, size=(HEIGHT, WIDTH, 3)).astype(np.uint8)
        imageio.imwrite(os.path.join(images_dir, f"img{i:03d}.png"), img)

    return data_dir


def test_parser_dataset_basic(synthetic_dataset):
    parser = Parser(
        data_dir=synthetic_dataset, factor=1, normalize=False, test_every=100
    )
    assert len(parser.image_names) == 4
    assert parser.points.shape[1] == 3

    dataset = Dataset(parser, split="train")
    item = dataset[0]
    assert item["image"].shape == (HEIGHT, WIDTH, 3)
    assert item["K"].shape == (3, 3)


def test_parser_colmap_dir_override(synthetic_dataset, tmp_path):
    """`colmap_dir` should let Parser read a reconstruction anywhere on disk,
    e.g. a `sparse/refined` copy written by bundle_adjustment.refine_reconstruction.
    """
    parser = Parser(
        data_dir=synthetic_dataset, factor=1, normalize=False, test_every=100
    )

    refined_dir = str(tmp_path / "sparse" / "refined")
    os.makedirs(refined_dir, exist_ok=True)
    pycolmap.Reconstruction(os.path.join(synthetic_dataset, "sparse", "0")).write(
        refined_dir
    )

    parser_refined = Parser(
        data_dir=synthetic_dataset,
        factor=1,
        normalize=False,
        test_every=100,
        colmap_dir=refined_dir,
    )
    assert len(parser_refined.image_names) == len(parser.image_names)


@pytest.mark.parametrize("dense_mode", ["augment", "replace"])
def test_parser_dense_points_path(synthetic_dataset, tmp_path, dense_mode):
    o3d = pytest.importorskip("open3d", reason="open3d not installed")

    parser = Parser(
        data_dir=synthetic_dataset, factor=1, normalize=False, test_every=100
    )
    n_sparse = parser.points.shape[0]

    rng = np.random.default_rng(2)
    dense_xyz = rng.uniform(-1, 1, size=(50, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_xyz)
    dense_ply = str(tmp_path / "dense.ply")
    o3d.io.write_point_cloud(dense_ply, pcd)

    parser_dense = Parser(
        data_dir=synthetic_dataset,
        factor=1,
        normalize=False,
        test_every=100,
        dense_points_path=dense_ply,
        dense_mode=dense_mode,
    )
    if dense_mode == "augment":
        assert parser_dense.points.shape[0] == n_sparse + 50
    else:
        assert parser_dense.points.shape[0] == 50


def test_dataset_mono_depth_dir(synthetic_dataset, tmp_path):
    parser = Parser(
        data_dir=synthetic_dataset, factor=1, normalize=False, test_every=100
    )

    mono_depth_dir = str(tmp_path / "mono_depth")
    os.makedirs(mono_depth_dir, exist_ok=True)
    rng = np.random.default_rng(3)
    for i in range(4):
        # Different resolution than the training image, to exercise resize.
        depth = rng.uniform(0.5, 5.0, size=(24, 32)).astype(np.float32)
        np.save(os.path.join(mono_depth_dir, f"img{i:03d}.npy"), depth)

    dataset = Dataset(parser, split="train", mono_depth_dir=mono_depth_dir)
    item = dataset[0]
    assert "mono_depth" in item
    assert item["mono_depth"].shape == item["image"].shape[:2]


def test_dataset_mono_depth_dir_with_patch_crop(synthetic_dataset, tmp_path):
    parser = Parser(
        data_dir=synthetic_dataset, factor=1, normalize=False, test_every=100
    )

    mono_depth_dir = str(tmp_path / "mono_depth")
    os.makedirs(mono_depth_dir, exist_ok=True)
    rng = np.random.default_rng(4)
    for i in range(4):
        depth = rng.uniform(0.5, 5.0, size=(HEIGHT, WIDTH)).astype(np.float32)
        np.save(os.path.join(mono_depth_dir, f"img{i:03d}.npy"), depth)

    dataset = Dataset(
        parser, split="train", patch_size=16, mono_depth_dir=mono_depth_dir
    )
    item = dataset[0]
    assert item["image"].shape[:2] == (16, 16)
    assert item["mono_depth"].shape == (16, 16)
