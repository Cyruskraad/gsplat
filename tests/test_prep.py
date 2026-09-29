# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture preparation: the geometry and radiometry glue around COLMAP."""

import math

import numpy

import pytest
import torch

from atlas.prep.geometry import (
    flash_positions,
    focus_point,
    normalise_scene,
    undistortion_grid,
    write_gaussian_ply,
)
from atlas.prep.images import (
    linear_to_srgb,
    list_images,
    read_exposure,
    resize_linear,
    srgb_to_linear,
)


def _look_at(centre, target=(0.0, 0.0, 0.0)):
    centre = torch.as_tensor(centre, dtype=torch.float64)
    forward = torch.nn.functional.normalize(
        torch.as_tensor(target, dtype=torch.float64) - centre, dim=0
    )
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, up), dim=0)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward])  # rows: camera axes in world
    viewmat = torch.eye(4, dtype=torch.float64)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ centre
    return viewmat


def _ring(count=8, radius=3.0, target=(0.4, -0.2, 0.1)):
    angles = torch.arange(count, dtype=torch.float64) * (2 * math.pi / count)
    centres = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles), 0.5 + 0 * angles],
        dim=-1,
    ) + torch.tensor(target, dtype=torch.float64)
    return torch.stack([_look_at(c, target) for c in centres]), centres


# --- geometry -------------------------------------------------------------------


def test_the_focus_is_where_the_optical_axes_meet():
    viewmats, centres = _ring()
    forward = viewmats[:, 2, :3]
    focus = focus_point(centres, forward)
    assert torch.allclose(focus, torch.tensor([0.4, -0.2, 0.1]).double(), atol=1e-9)


def test_parallel_axes_do_not_produce_a_point_at_infinity():
    centres = torch.tensor([[x, 0.0, 0.0] for x in range(5)], dtype=torch.float64)
    forward = torch.tensor([[0.0, 0.0, 1.0]] * 5, dtype=torch.float64)
    focus = focus_point(centres, forward)
    assert torch.isfinite(focus).all() and float(focus[2]) > 0


def test_normalisation_is_a_similarity_that_preserves_every_projection():
    viewmats, _ = _ring()
    focus = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float64)
    normalised, scale = normalise_scene(viewmats, focus, camera_distance=1.0)
    point = torch.tensor([0.7, 0.1, -0.3], dtype=torch.float64)
    moved = scale * (point - focus)

    def project(viewmat, x):
        cam = viewmat[:3, :3] @ x + viewmat[:3, 3]
        return cam[:2] / cam[2]

    for before, after in zip(viewmats, normalised):
        assert torch.allclose(project(before, point), project(after, moved), atol=1e-12)
    centres = -(
        normalised[:, :3, :3].transpose(-1, -2) @ normalised[:, :3, 3:]
    ).squeeze(-1)
    assert torch.allclose(centres.norm(dim=-1).median(), torch.tensor(1.0).double())


def test_the_flash_rides_on_the_camera():
    viewmats, centres = _ring()
    assert torch.allclose(flash_positions(viewmats, (0, 0, 0)), centres, atol=1e-12)
    offset = (0.05, -0.12, 0.0)
    lights = flash_positions(viewmats, offset)
    for viewmat, light in zip(viewmats, lights):
        in_camera = viewmat[:3, :3] @ light + viewmat[:3, 3]
        assert torch.allclose(in_camera, torch.tensor(offset).double(), atol=1e-12)


def test_undistorting_a_pinhole_is_the_identity():
    pycolmap = pytest.importorskip("pycolmap")
    camera = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 100.0, 40, 30)
    grid = undistortion_grid(camera, camera)
    xs = (torch.arange(40, dtype=torch.float64) + 0.5) / 40 * 2 - 1
    assert torch.allclose(grid[0, :, 0], xs, atol=1e-9)
    ys = (torch.arange(30, dtype=torch.float64) + 0.5) / 30 * 2 - 1
    assert torch.allclose(grid[:, 0, 1], ys, atol=1e-9)


def test_the_initialisation_ply_loads_as_a_relight_model(tmp_path):
    from atlas.model import RelightSplats

    means = torch.randn(50, 3)
    colours = torch.rand(50, 3)
    write_gaussian_ply(tmp_path / "init.ply", means, colours)
    model = RelightSplats.from_ply(tmp_path / "init.ply", num_atoms=4)
    assert model.num_primitives == 50
    assert torch.allclose(model.means.float(), means, atol=1e-6)
    assert torch.isfinite(model.scales).all()


# --- images -----------------------------------------------------------------------


def test_the_srgb_curve_round_trips():
    x = torch.linspace(0, 1, 1001, dtype=torch.float64)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12)


def test_downsampling_happens_in_linear_space_and_keeps_the_mean():
    image = torch.zeros(40, 60, 3)
    image[:, ::2] = 1.0
    small = resize_linear(image, 30)
    assert small.shape == (20, 30, 3)
    assert abs(float(small.mean()) - 0.5) < 1e-3


def test_raw_wins_over_its_own_jpeg(tmp_path):
    for name in ("a.JPG", "a.NEF", "b.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.NEF", "b.jpg"]


def test_the_flash_exposure_comes_from_iso_and_aperture(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    image = PIL.new("RGB", (4, 4))
    exif = image.getexif()
    detail = exif.get_ifd(0x8769)
    detail[0x8827] = 400
    detail[0x829D] = 8.0
    detail[0x829A] = 1 / 60
    detail[0x9209] = 1
    image.save(tmp_path / "a.jpg", exif=exif)
    info = read_exposure(tmp_path / "a.jpg")
    assert info.flash_fired is True
    assert info.flash_gain() == pytest.approx(400 / 64)


def test_jpeg_decodes_without_opencv(tmp_path, monkeypatch):
    PIL = pytest.importorskip("PIL.Image")
    import atlas.imageio as imageio
    from atlas.prep.images import decode_image

    PIL.new("RGB", (8, 6), (128, 128, 128)).save(tmp_path / "a.jpg", quality=100)
    monkeypatch.setattr(imageio, "opencv_available", lambda: False)
    decoded = decode_image(tmp_path / "a.jpg")
    assert decoded.linear.shape == (6, 8, 3)
    assert not decoded.linear_exact
    assert (
        abs(
            float(decoded.linear.mean())
            - float(srgb_to_linear(torch.tensor(128 / 255)))
        )
        < 0.01
    )


# --- the whole pipeline -----------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_photos(tmp_path_factory):
    """Twelve photographs of a scene COLMAP reconstructs reliably."""
    pycolmap = pytest.importorskip("pycolmap")
    pycolmap.set_random_seed(0)
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = 12
    options.num_points3D = 400
    options.camera_width, options.camera_height = 640, 480
    reconstruction = pycolmap.synthesize_dataset(options)
    folder = tmp_path_factory.mktemp("photos")
    pycolmap.synthesize_images(pycolmap.SyntheticImageOptions(), reconstruction, folder)
    return folder, reconstruction


def test_photographs_become_a_capture_the_loader_accepts(synthetic_photos, tmp_path):
    from atlas.data.loader import load_capture
    from atlas.model import RelightSplats
    from atlas.prep import PrepConfig, SfMConfig, prepare_capture

    folder, truth = synthetic_photos
    config = PrepConfig(
        sfm=SfMConfig(features="sift", matcher="bruteforce", device="cpu"),
        flash_offset=(0.0, -0.1, 0.0),
    )
    report = prepare_capture(folder, tmp_path / "cap", config)
    assert report["sfm"]["num_registered"] == 12
    assert report["sfm"]["mean_reprojection_error_px"] < 1.5

    capture = load_capture(tmp_path / "cap")
    assert len(capture) == 12
    true_focal = truth.cameras[1].params[0]
    assert float(capture.intrinsics[0, 0]) == pytest.approx(true_focal, rel=0.02)

    coupling = capture.coupling_report()
    assert coupling.verdict == "co_located"

    # Recovered camera layout matches the truth up to a similarity.
    names = [f.image_path.name for f in capture.frames]
    truth_centres = {im.name: im.projection_center() for im in truth.images.values()}
    a = torch.from_numpy(numpy.array([truth_centres[n] for n in names]))
    b = capture.camera_positions()
    da, db = torch.cdist(a, a), torch.cdist(b, b)
    ratio = db[da > 0] / da[da > 0]
    assert float(ratio.std() / ratio.mean()) < 0.02

    image = capture.image(0)
    assert image.shape == (capture.height, capture.width, 3)
    assert float(image.max()) > 0
    RelightSplats.from_ply(tmp_path / "cap" / "init.ply", num_atoms=4)


def test_learned_features_fall_back_to_sift_with_a_note(synthetic_photos, tmp_path):
    from atlas.prep import SfMConfig, learned_features_available, run_sfm

    if learned_features_available():
        pytest.skip("this pycolmap has ONNX support; the fallback is not exercised")
    folder, _ = synthetic_photos
    result = run_sfm(folder, tmp_path, SfMConfig(features="aliked", device="cpu"))
    assert result.features_used == "sift"
    assert any("unavailable" in note for note in result.notes)
    assert result.num_registered == 12
