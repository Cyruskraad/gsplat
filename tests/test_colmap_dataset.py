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
import torch

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
cv2 = pytest.importorskip("cv2", reason="opencv not installed")
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


def test_dataset_mono_depth_dir_stays_aligned_under_distortion(tmp_path):
    """A distorted camera's mono_depth must go through the same undistortion
    remap + ROI crop as the image, not just a plain resize -- otherwise the
    two become spatially misaligned. Build `mono_depth` as an exact copy of
    the image's own red channel; since both then go through the identical
    cv2.remap/crop, they should end up numerically matching.
    """
    data_dir = str(tmp_path)
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    width, height = 80, 60
    recon = pycolmap.Reconstruction()
    # SIMPLE_RADIAL params: [f, cx, cy, k] -- a non-trivial k so undistortion
    # actually warps content, not just crops it.
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_RADIAL, 60.0, width, height
    )
    camera.params = np.array([60.0, width / 2, height / 2, -0.25], dtype=np.float64)
    recon.add_camera_with_trivial_rig(camera)

    img = pycolmap.Image()
    img.image_id = 1
    img.camera_id = 1
    img.name = "img000.png"
    img.points2D = []
    cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3))
    recon.add_image_with_trivial_frame(img, cam_from_world)

    os.makedirs(sparse_dir, exist_ok=True)
    recon.write(sparse_dir)

    rng = np.random.default_rng(5)
    image = rng.uniform(0, 255, size=(height, width, 3)).astype(np.uint8)
    imageio.imwrite(os.path.join(images_dir, "img000.png"), image)

    mono_depth_dir = str(tmp_path / "mono_depth")
    os.makedirs(mono_depth_dir, exist_ok=True)
    # Exact copy of the red channel -- same content, so the two should
    # receive numerically matching geometric warps.
    np.save(
        os.path.join(mono_depth_dir, "img000.npy"), image[..., 0].astype(np.float32)
    )

    parser = Parser(data_dir=data_dir, factor=1, normalize=False, test_every=100)
    assert len(parser.params_dict[1]) > 0, "camera should have distortion params"

    dataset = Dataset(parser, split="val", mono_depth_dir=mono_depth_dir)
    item = dataset[0]

    image_red = item["image"][..., 0].numpy()
    mono = item["mono_depth"].numpy()
    assert image_red.shape == mono.shape
    np.testing.assert_allclose(mono, image_red, atol=2.0)


def test_dataset_mask_dir_basic(synthetic_dataset, tmp_path):
    """A mask_dir PNG (nonzero = keep, 0 = exclude) should come back in
    `data["mask"]` as a matching-shape bool tensor, True where nonzero.
    """
    parser = Parser(
        data_dir=synthetic_dataset, factor=1, normalize=False, test_every=100
    )

    mask_dir = str(tmp_path / "masks")
    os.makedirs(mask_dir, exist_ok=True)
    for i in range(4):
        mask = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        mask[:, : WIDTH // 2] = 0  # left half excluded ("transient")
        imageio.imwrite(os.path.join(mask_dir, f"img{i:03d}.png"), mask)

    dataset = Dataset(parser, split="train", mask_dir=mask_dir)
    item = dataset[0]
    assert "mask" in item
    assert item["mask"].shape == item["image"].shape[:2]
    assert item["mask"].dtype == torch.bool
    assert not item["mask"][:, : WIDTH // 2].any()
    assert item["mask"][:, WIDTH // 2 :].all()


def test_dataset_mask_dir_with_patch_crop(synthetic_dataset, tmp_path):
    """The mask must be cropped in step with `image` under patch_size, the
    same as mono_depth -- otherwise the two land at different shapes/content.
    """
    parser = Parser(
        data_dir=synthetic_dataset, factor=1, normalize=False, test_every=100
    )

    mask_dir = str(tmp_path / "masks")
    os.makedirs(mask_dir, exist_ok=True)
    for i in range(4):
        mask = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        mask[:, : WIDTH // 2] = 0
        imageio.imwrite(os.path.join(mask_dir, f"img{i:03d}.png"), mask)

    dataset = Dataset(parser, split="train", patch_size=16, mask_dir=mask_dir)
    item = dataset[0]
    assert item["image"].shape[:2] == (16, 16)
    assert item["mask"].shape == (16, 16)


def test_dataset_mask_dir_stays_aligned_under_distortion(tmp_path):
    """Same alignment contract as mono_depth: a mask must undergo the same
    undistortion remap + ROI crop as the image, not a plain resize. Build the
    mask as a binarized copy of the image's own red channel (thresholded),
    then re-derive the same threshold from the *undistorted* image and check
    they match -- if the mask took a different geometric path, they wouldn't.
    """
    data_dir = str(tmp_path)
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    width, height = 80, 60
    recon = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_RADIAL, 60.0, width, height
    )
    camera.params = np.array([60.0, width / 2, height / 2, -0.25], dtype=np.float64)
    recon.add_camera_with_trivial_rig(camera)

    img = pycolmap.Image()
    img.image_id = 1
    img.camera_id = 1
    img.name = "img000.png"
    img.points2D = []
    cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3))
    recon.add_image_with_trivial_frame(img, cam_from_world)

    os.makedirs(sparse_dir, exist_ok=True)
    recon.write(sparse_dir)

    rng = np.random.default_rng(6)
    image = rng.uniform(0, 255, size=(height, width, 3)).astype(np.uint8)
    imageio.imwrite(os.path.join(images_dir, "img000.png"), image)

    mask_dir = str(tmp_path / "masks")
    os.makedirs(mask_dir, exist_ok=True)
    rng_mask = np.random.default_rng(9)
    original_mask = (rng_mask.uniform(0, 1, size=(height, width)) > 0.5).astype(
        np.uint8
    ) * 255
    imageio.imwrite(os.path.join(mask_dir, "img000.png"), original_mask)

    parser = Parser(data_dir=data_dir, factor=1, normalize=False, test_every=100)
    assert len(parser.params_dict[1]) > 0, "camera should have distortion params"

    dataset = Dataset(parser, split="val", mask_dir=mask_dir)
    item = dataset[0]

    # Independently reproduce the expected warp from the parser's own
    # undistortion maps -- exactly the transform `image` itself goes
    # through -- rather than relying on any numerical coincidence with the
    # (bilinearly-interpolated) image channel.
    mapx, mapy = parser.mapx_dict[1], parser.mapy_dict[1]
    x, y, w, h = parser.roi_undist_dict[1]
    expected_mask = cv2.remap(
        (original_mask != 0).astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST
    )
    expected_mask = expected_mask[y : y + h, x : x + w].astype(bool)

    np.testing.assert_array_equal(item["mask"].numpy(), expected_mask)


def test_dataset_mask_dir_combines_with_fisheye_roi(tmp_path):
    """When both the fisheye ROI mask (from Parser) and a mask_dir transient
    mask are present, `data["mask"]` should be their logical AND.
    """
    data_dir = str(tmp_path)
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    width, height = 80, 60
    recon = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.OPENCV_FISHEYE, 40.0, width, height
    )
    # A non-zero k1 (unlike an all-zero-distortion fisheye, which reduces to
    # an identity mapping) so the undistortion maps actually push some
    # border pixels out of the source image, producing a genuine
    # (non-all-True) ROI mask.
    camera.params = np.array(
        [40.0, 40.0, width / 2, height / 2, 0.5, 0.0, 0.0, 0.0], dtype=np.float64
    )
    recon.add_camera_with_trivial_rig(camera)

    img = pycolmap.Image()
    img.image_id = 1
    img.camera_id = 1
    img.name = "img000.png"
    img.points2D = []
    cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3))
    recon.add_image_with_trivial_frame(img, cam_from_world)

    os.makedirs(sparse_dir, exist_ok=True)
    recon.write(sparse_dir)

    rng = np.random.default_rng(7)
    image = rng.uniform(0, 255, size=(height, width, 3)).astype(np.uint8)
    imageio.imwrite(os.path.join(images_dir, "img000.png"), image)

    parser = Parser(data_dir=data_dir, factor=1, normalize=False, test_every=100)
    fisheye_roi_mask = parser.mask_dict[1]
    assert fisheye_roi_mask is not None, "fisheye camera should produce an ROI mask"
    assert not fisheye_roi_mask.all(), "ROI mask should exclude some border pixels"

    out_h, out_w = fisheye_roi_mask.shape
    mask_dir = str(tmp_path / "masks")
    os.makedirs(mask_dir, exist_ok=True)
    # An all-True transient mask at the *original* resolution -- after
    # undergoing the same undistortion remap+crop as the image, it should
    # end up all-True too, so the combination equals the ROI mask exactly.
    all_keep = np.full((height, width), 255, dtype=np.uint8)
    imageio.imwrite(os.path.join(mask_dir, "img000.png"), all_keep)

    dataset = Dataset(parser, split="val", mask_dir=mask_dir)
    item = dataset[0]
    assert item["mask"].shape == (out_h, out_w)
    # Should match the ROI mask almost everywhere -- nearest-neighbor
    # remap's rounding can disagree with the parser's strict in-bounds
    # inequalities on a handful of boundary pixels.
    mismatch_frac = (item["mask"].numpy() != fisheye_roi_mask).mean()
    assert mismatch_frac < 0.05, f"combined mask diverged: {mismatch_frac:.3f}"
    # And it should genuinely be a combination, not just one input passed
    # through: some pixels must be excluded (from the ROI mask).
    assert not item["mask"].numpy().all()


def test_dataset_loads_masks_in_the_documented_recipe_format(
    synthetic_dataset, tmp_path
):
    """A mask written exactly the way `docs/photogrammetry.md`'s Mask R-CNN
    recipe writes it must load.

    The recipe's last step is
    `Image.fromarray((keep_mask * 255).astype(np.uint8)).save(...)` -- a
    single-channel PIL PNG built from a *bool* array, which is a different
    writer and a different code path from the `imageio` masks the other tests
    use. Documented instructions that silently stop producing loadable output
    are worse than none, so pin the recipe's actual output format here.
    """
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")

    parser = Parser(
        data_dir=synthetic_dataset, factor=1, normalize=False, test_every=100
    )

    mask_dir = str(tmp_path / "recipe_masks")
    os.makedirs(mask_dir, exist_ok=True)
    # `keep_mask` as the recipe builds it: True = keep, cleared over the
    # pixels a detected "movable" instance covers.
    keep_mask = np.ones((HEIGHT, WIDTH), dtype=bool)
    keep_mask[10:30, 10:40] = False
    for i in range(4):
        Image.fromarray((keep_mask * 255).astype(np.uint8)).save(
            os.path.join(mask_dir, f"img{i:03d}.png")
        )

    dataset = Dataset(parser, split="train", mask_dir=mask_dir)
    item = dataset[0]

    assert item["mask"].dtype == torch.bool
    assert item["mask"].shape == item["image"].shape[:2]
    # The excluded block lands exactly where the recipe put it, at the same
    # scale -- not mirrored, transposed or resized.
    assert not item["mask"][10:30, 10:40].any()
    assert item["mask"][0:10, :].all()
    excluded_fraction = float((~item["mask"]).float().mean())
    assert excluded_fraction == pytest.approx((20 * 30) / (HEIGHT * WIDTH))


def test_dataset_squeezes_and_rejects_non_2d_mono_depth(synthetic_dataset, tmp_path):
    """A depth map with a leading batch axis must load correctly, not silently
    become garbage; anything genuinely not 2D must raise.

    Depth models commonly emit `(1, H, W)` -- a transformers depth-estimation
    pipeline's `predicted_depth` among them, which is what
    `docs/photogrammetry.md`'s recipe produces. Before this was handled, such a
    map sailed through the resize step: `cv2.resize` reads a `(1, H, W)` array
    as a one-row image with W channels and returns `(H, W, W)`, so training
    would have been supervised against reshaped noise with no error raised.
    """
    parser = Parser(
        data_dir=synthetic_dataset, factor=1, normalize=False, test_every=100
    )

    rng = np.random.default_rng(11)
    depth_2d = rng.uniform(1.0, 5.0, size=(HEIGHT, WIDTH)).astype(np.float32)

    # (1, H, W): squeezed, and identical to the same map saved as (H, W).
    batched_dir = str(tmp_path / "mono_depth_batched")
    plain_dir = str(tmp_path / "mono_depth_plain")
    os.makedirs(batched_dir, exist_ok=True)
    os.makedirs(plain_dir, exist_ok=True)
    for i in range(4):
        np.save(os.path.join(batched_dir, f"img{i:03d}.npy"), depth_2d[None])
        np.save(os.path.join(plain_dir, f"img{i:03d}.npy"), depth_2d)

    batched = Dataset(parser, split="train", mono_depth_dir=batched_dir)[0]
    plain = Dataset(parser, split="train", mono_depth_dir=plain_dir)[0]
    assert batched["mono_depth"].shape == plain["image"].shape[:2]
    np.testing.assert_allclose(
        batched["mono_depth"].numpy(), plain["mono_depth"].numpy()
    )

    # (H, W, 3) can't be squeezed into a depth map: refuse it, and say which
    # file is at fault rather than failing somewhere downstream.
    bad_dir = str(tmp_path / "mono_depth_bad")
    os.makedirs(bad_dir, exist_ok=True)
    for i in range(4):
        np.save(
            os.path.join(bad_dir, f"img{i:03d}.npy"),
            rng.uniform(0, 1, size=(HEIGHT, WIDTH, 3)).astype(np.float32),
        )

    with pytest.raises(ValueError, match=r"img\d+\.npy.*not a single \(H, W\)"):
        Dataset(parser, split="train", mono_depth_dir=bad_dir)[0]
