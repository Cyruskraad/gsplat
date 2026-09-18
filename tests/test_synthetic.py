# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
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

"""The synthetic capture, and the three questions only a known answer settles.

1. Does the manifest say what the renderer did? The pose convention is the
   likeliest thing in the whole pipeline to be subtly wrong, because getting it
   wrong flips the object and still looks like an object.
2. Does ``solve_flash_offset`` recover an offset that was **planted**? Until now
   it has only been checked against fixtures built by the same reasoning it
   uses.
3. Do the two held-out splits actually differ? This is where the capture
   protocol turns out to matter, and the answer is measured here rather than
   argued in a document.
"""

import dataclasses
import json
import math

import pytest

torch = pytest.importorskip("torch")

from atlas.data.synthetic import (  # noqa: E402
    SyntheticConfig,
    fit_lobe_transport,
    generate_capture,
    nerf_to_viewmat,
    viewmat_to_nerf,
)
from atlas.functional.atoms import (
    evaluate_atoms,
    fibonacci_sphere,
    make_sg_atoms,
)  # noqa: E402
from atlas.functional.calibration import solve_flash_offset  # noqa: E402
from atlas.functional.splits import split_lights, split_views  # noqa: E402
from atlas.imageio import read_png  # noqa: E402
from atlas.reference import render_reference  # noqa: E402

SMALL = SyntheticConfig(
    num_views=4, num_lights=4, num_primitives=48, width=32, height=32, num_atoms=12
)


# --- the pose convention ----------------------------------------------------


def test_the_nerf_transform_round_trips():
    """A flipped ``y`` renders an upside-down object that still looks like an
    object, so this conversion has to be checked and not eyeballed."""
    from atlas.reference import look_at

    viewmat = look_at(torch.tensor([1.0, -2.0, 0.7]), torch.tensor([0.1, 0.0, -0.2]))
    assert torch.allclose(
        nerf_to_viewmat(viewmat_to_nerf(viewmat)), viewmat, atol=1e-12
    )


def test_the_nerf_transform_matrix_holds_the_camera_centre_in_its_last_column():
    from atlas.reference import look_at

    eye = torch.tensor([1.0, -2.0, 0.7], dtype=torch.float64)
    transform = viewmat_to_nerf(look_at(eye, torch.zeros(3)))
    assert torch.allclose(transform[:3, 3], eye, atol=1e-12)


def test_the_nerf_matrix_columns_are_right_up_and_backward():
    """The absolute check, because the round trip is only a relative one.

    ``viewmat_to_nerf`` and ``nerf_to_viewmat`` share one constant, so they
    round-trip perfectly however wrong that constant is -- flipping x instead
    of y would pass every other test here. A NeRF ``transform_matrix`` holds
    the camera axes as columns in OpenGL order: right, **up**, **backward**.

    For a camera at ``(0, -3, 0)`` looking at the origin with ``+z`` up, those
    are ``(1, 0, 0)``, ``(0, 0, 1)`` and ``(0, -1, 0)``.
    """
    from atlas.reference import look_at

    transform = viewmat_to_nerf(look_at(torch.tensor([0.0, -3.0, 0.0]), torch.zeros(3)))
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    assert torch.allclose(transform[:3, :3], expected, atol=1e-12), transform[
        :3, :3
    ].tolist()


def test_the_axis_flip_is_a_real_flip_and_not_the_identity():
    """The premise for the round-trip test: if ``_AXIS_FLIP`` were the identity
    the conversion would round-trip perfectly and mean nothing."""
    from atlas.reference import look_at

    viewmat = look_at(torch.tensor([0.0, -3.0, 0.0]), torch.zeros(3))
    assert not torch.allclose(
        viewmat_to_nerf(viewmat), torch.linalg.inv(viewmat), atol=1e-6
    )


def test_a_frame_rendered_from_the_manifest_reproduces_the_image_that_was_written(
    tmp_path,
):
    """The whole write-then-read path, through the manifest rather than around
    it: poses taken from ``transform_matrix``, converted back, re-rendered."""
    capture = generate_capture(tmp_path / "cap", SMALL)
    manifest = json.loads((tmp_path / "cap" / "transforms.json").read_text())
    from atlas.functional.atoms import project_point_light
    from atlas.functional.nearfield import incident_radiance
    from atlas.functional.transport import contract

    for record in manifest["frames"][:3]:
        viewmat = nerf_to_viewmat(torch.tensor(record["transform_matrix"]))
        light = torch.tensor(record["atlas"]["light_position"], dtype=torch.float64)
        intensity = torch.tensor(
            record["atlas"]["light_intensity"], dtype=torch.float64
        )
        directions, radiance = incident_radiance(
            capture.means,
            light,
            intensity,
            reference_distance=record["atlas"]["reference_distance"],
        )
        colors = contract(
            capture.transport,
            project_point_light(
                directions, radiance, capture.atom_axes, capture.atom_sharpness
            ),
        ).clamp_min(0.0)
        image, _ = render_reference(
            capture.means,
            capture.quats,
            capture.log_scales,
            capture.opacity_logits,
            colors,
            viewmat,
            capture.intrinsics,
            SMALL.width,
            SMALL.height,
        )
        written = read_png(tmp_path / "cap" / record["file_path"]).double() / 255.0
        expected = (image / SMALL.scale).clamp(0.0, 1.0)
        # 8-bit storage, so agreement is to half a code value and no closer.
        assert float((written - expected).abs().max()) <= 1.0 / 255.0 + 1e-9


# --- the surface response ---------------------------------------------------


def _fit_residual(num_atoms, *, shininess=None, num_normals=8):
    """Worst and rms error of the fitted transport against the lobe it fitted,
    both relative to the lobe's own peak."""
    normals = fibonacci_sphere(num_normals).to(torch.float64)
    axes, sharpnesses = make_sg_atoms(num_atoms)
    axes, sharpnesses = axes.double(), sharpnesses.double()
    probe = fibonacci_sphere(256).to(torch.float64)
    weights = evaluate_atoms(probe, axes, sharpnesses).double()
    cosine = (normals @ probe.T).clamp_min(0.0)

    if shininess is None:
        albedo = torch.full((num_normals, 3), 0.6, dtype=torch.float64)
        transport = fit_lobe_transport(normals, albedo, axes, sharpnesses, specular=0.0)
        truth = cosine.unsqueeze(-1) * (albedo / math.pi).unsqueeze(1)
    else:
        albedo = torch.zeros(num_normals, 3, dtype=torch.float64)
        transport = fit_lobe_transport(
            normals, albedo, axes, sharpnesses, specular=1.0, shininess=shininess
        )
        truth = cosine.pow(shininess).unsqueeze(-1).expand(-1, -1, 3)

    error = torch.einsum("ncb,sb->nsc", transport, weights) - truth
    peak = truth.abs().max()
    return float(error.abs().max() / peak), float(error.pow(2).mean().sqrt() / peak)


def test_the_fitted_transport_reproduces_a_broad_lobe_and_improves_with_more_atoms():
    """Measured worst-case relative error of the fit, over 256 probe directions:

        atoms   diffuse   p = 2   p = 20   p = 60
           12    0.136    0.218    0.828    0.937
           32    0.078    0.039    0.478    0.742
           64    0.055    0.017    0.265    0.631

    The worst case sits at the terminator, where ``max(0, n . w)`` has a kink
    that no smooth basis reproduces; the rms error at 32 atoms is 0.020.
    """
    worst_12, _ = _fit_residual(12)
    worst_32, rms_32 = _fit_residual(32)
    worst_64, _ = _fit_residual(64)
    assert worst_12 > worst_32 > worst_64, (worst_12, worst_32, worst_64)
    assert worst_32 < 0.10, worst_32
    assert rms_32 < 0.03, rms_32


def test_a_narrow_lobe_does_not_fit_a_small_basis_and_that_is_the_point():
    """The knob that makes the gate fail on purpose.

    A backscatter lobe of exponent 60 is far narrower than twelve spherical
    Gaussians can represent. At 12 atoms it fits seven times worse than the
    diffuse lobe does, and -- the part that matters -- quadrupling the atom
    count barely helps it, so a model trained on such a capture has no way to
    predict an unseen light.
    """
    broad, _ = _fit_residual(12)
    narrow_12, _ = _fit_residual(12, shininess=60.0)
    narrow_64, _ = _fit_residual(64, shininess=60.0)

    assert narrow_12 > 0.8, narrow_12
    assert narrow_12 > 5 * broad, (narrow_12, broad)
    # More atoms rescue the broad lobe far better than the narrow one: 2.5x
    # against 1.5x, measured.
    broad_64, _ = _fit_residual(64)
    assert (broad / broad_64) > 1.5 * (narrow_12 / narrow_64)


def test_an_underdetermined_fit_is_refused_rather_than_arbitrary():
    axes, sharpnesses = make_sg_atoms(32)
    with pytest.raises(ValueError, match="at least the atom count"):
        fit_lobe_transport(
            torch.zeros(2, 3, dtype=torch.float64),
            torch.zeros(2, 3, dtype=torch.float64),
            axes.double(),
            sharpnesses.double(),
            num_samples=8,
        )


# --- the finding: the capture protocol decides whether the gate can fail ----


def test_a_free_flash_gives_two_splits_that_select_different_shots():
    capture = generate_capture(
        pytest.importorskip("tempfile").mkdtemp() + "/free", SMALL, write_images=False
    )
    assert capture.splits_are_independent
    views = sorted({f["atlas"]["view_index"] for f in capture.frames})
    lights = sorted({f["atlas"]["light_index"] for f in capture.frames})
    assert len(capture.frames) == len(views) * len(lights)


def test_a_bracket_flash_collapses_the_two_splits_onto_each_other(tmp_path):
    """The measured reason the capture protocol matters.

    With the flash rigidly attached to the camera, the light is a function of
    the camera. Holding out a light holds out its view, the two reported
    numbers are the same number, and the gate passes whatever the model
    learned. A gate that cannot fail is not a gate, which is why the manifest
    records ``splits_are_independent`` rather than leaving it to be noticed.
    """
    config = dataclasses.replace(SMALL, flash_mode="bracket")
    capture = generate_capture(tmp_path / "bracket", config, write_images=False)
    assert not capture.splits_are_independent

    # One shot per view, and its light index is its view index.
    assert len(capture.frames) == config.num_views
    assert all(
        f["atlas"]["view_index"] == f["atlas"]["light_index"] for f in capture.frames
    )

    # The splits, run for real, select the same shots.
    centres = torch.stack([-v[:3, :3].T @ v[:3, 3] for v in capture.viewmats])
    directions = capture.light_positions / capture.light_positions.norm(
        dim=-1, keepdim=True
    )
    held_views = set(split_views(centres, 1, 1).test.tolist())
    held_lights = set(split_lights(directions, 1, 1).test.tolist())
    assert held_views == held_lights, (held_views, held_lights)

    manifest = json.loads((tmp_path / "bracket" / "transforms.json").read_text())
    assert manifest["atlas"]["splits_are_independent"] is False


def test_the_manifest_says_which_mode_it_was_shot_in(tmp_path):
    capture = generate_capture(tmp_path / "cap", SMALL, write_images=False)
    manifest = json.loads((tmp_path / "cap" / "transforms.json").read_text())
    assert manifest["atlas"]["flash_mode"] == "free"
    assert manifest["atlas"]["splits_are_independent"] is True
    assert manifest["atlas"]["colour_space"] == "linear"
    assert manifest["atlas"]["config"]["seed"] == SMALL.seed
    assert capture.splits_are_independent


def test_an_unknown_flash_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="'free' or 'bracket'"):
        generate_capture(
            tmp_path / "cap",
            dataclasses.replace(SMALL, flash_mode="ringlight"),
            write_images=False,
        )


# --- calibration against a planted answer ----------------------------------


def test_the_planted_bracket_offset_is_recovered_from_perfect_rays(tmp_path):
    """The first end-to-end check of the calibration work against a known
    answer. Every previous test of it built its fixtures from the same geometry
    the solver assumes; this one takes the offset the *generator* used.
    """
    config = dataclasses.replace(SMALL, flash_mode="bracket", num_views=12)
    capture = generate_capture(tmp_path / "cap", config, write_images=False)

    centres = torch.stack([-v[:3, :3].T @ v[:3, 3] for v in capture.viewmats])
    rotations = torch.stack([v[:3, :3].T for v in capture.viewmats])  # camera -> world

    # A perfect "chrome sphere" observation: a ray through the true light from
    # an arbitrary but well-spread vantage point.
    generator = torch.Generator().manual_seed(7)
    vantage = fibonacci_sphere(config.num_views).to(torch.float64) * 2.5
    offsets = capture.light_positions - vantage
    directions = offsets / offsets.norm(dim=-1, keepdim=True)

    solution = solve_flash_offset(centres, rotations, vantage, directions)
    planted = torch.tensor(config.flash_offset, dtype=torch.float64)
    error = float((solution.offset - planted).norm())
    assert error < 1e-6, (error, solution.offset.tolist(), planted.tolist())
    assert float(solution.condition) < 1e3, float(solution.condition)


def test_the_orbit_varies_all_three_of_the_things_that_condition_the_fit(tmp_path):
    """Standoff, elevation and roll, which is the generator's own obligation.

    The *consequence* -- that a perfectly regular orbit leaves the offset fit
    rank-deficient -- is pinned in ``tests/test_calibration.py`` against a
    fixture that solves the reflection properly. Repeating it here with a
    simplified ray model measured a condition number of 1.1 for both orbits and
    therefore tested nothing, so this checks the input property instead.
    """
    config = dataclasses.replace(SMALL, num_views=16)
    capture = generate_capture(tmp_path / "cap", config, write_images=False)
    centres = torch.stack([-v[:3, :3].T @ v[:3, 3] for v in capture.viewmats])

    standoff = centres.norm(dim=-1)
    elevation = torch.asin((centres[:, 2] / standoff).clamp(-1.0, 1.0))
    # The camera's own up axis against world up: roll.
    roll = torch.stack(
        [
            v[:3, :3][1] @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
            for v in capture.viewmats
        ]
    )

    assert float(standoff.std() / standoff.mean()) > 0.03, float(standoff.std())
    assert float(elevation.max() - elevation.min()) > math.radians(45.0)
    assert float(roll.std()) > 0.05, float(roll.std())


# --- determinism and guards -------------------------------------------------


def test_the_same_seed_gives_the_same_capture(tmp_path):
    first = generate_capture(tmp_path / "a", SMALL, write_images=False)
    second = generate_capture(tmp_path / "b", SMALL, write_images=False)
    assert torch.equal(first.means, second.means)
    assert torch.equal(first.transport, second.transport)
    assert torch.equal(first.viewmats, second.viewmats)


def test_a_different_seed_gives_a_different_capture(tmp_path):
    first = generate_capture(tmp_path / "a", SMALL, write_images=False)
    second = generate_capture(
        tmp_path / "b", dataclasses.replace(SMALL, seed=99), write_images=False
    )
    assert not torch.equal(first.means, second.means)


def test_generating_over_an_existing_capture_is_refused(tmp_path):
    generate_capture(tmp_path / "cap", SMALL, write_images=False)
    with pytest.raises(FileExistsError, match="mixture of two"):
        generate_capture(tmp_path / "cap", SMALL, write_images=False)


def test_masks_are_written_and_cover_the_object(tmp_path):
    capture = generate_capture(tmp_path / "cap", SMALL)
    record = capture.frames[0]["atlas"]
    mask = read_png(tmp_path / "cap" / record["mask_path"]).double() / 255.0
    assert mask.shape == (SMALL.height, SMALL.width, 1)
    coverage = float((mask > 0.5).double().mean())
    assert 0.05 < coverage < 0.9, coverage


def test_the_ground_truth_is_saved_beside_the_images(tmp_path):
    capture = generate_capture(tmp_path / "cap", SMALL, write_images=False)
    saved = torch.load(tmp_path / "cap" / "ground_truth.pt", weights_only=False)
    assert torch.equal(saved["transport"], capture.transport)
    assert saved["config"]["num_atoms"] == SMALL.num_atoms


def test_the_illumination_varies_enough_for_the_inspector_to_call_it_relighting(
    tmp_path,
):
    """The inspector's own thresholds: the brightest pixel must move by at least
    5% of the frame, or the mean luma must vary by at least 15%. Measured here
    so the fixture cannot drift below them unnoticed."""
    from atlas.tools.inspect_capture import HIGHLIGHT_MOVING, LUMA_VARIES

    config = dataclasses.replace(SMALL, width=64, height=64)
    capture = generate_capture(tmp_path / "cap", config)
    images = [
        read_png(tmp_path / "cap" / f["file_path"]).double().mean(-1) / 255.0
        for f in capture.frames
    ]
    lumas = torch.tensor([float(i.mean()) for i in images])
    spread = float((lumas.max() - lumas.min()) / lumas.mean())
    peaks = torch.stack([(i == i.max()).nonzero().double().mean(0) for i in images])
    motion = float(peaks.std(dim=0).norm()) / config.width
    assert spread > LUMA_VARIES, spread
    assert motion > HIGHLIGHT_MOVING, motion


# --- the inspector, run against a capture for the first time ---------------


def test_the_inspector_calls_the_generated_capture_usable(tmp_path):
    """T1 has only ever been run against fixtures it built itself. This is the
    first time it meets a capture produced by a different part of the project,
    and it is the closest thing available to a rehearsal of the real one.
    """
    pytest.importorskip("PIL", reason="the inspector reads images through Pillow")
    from atlas.tools.inspect_capture import inspect_capture

    config = dataclasses.replace(SMALL, num_views=6, num_lights=6, width=64, height=64)
    generate_capture(tmp_path / "cap", config)
    report = inspect_capture(tmp_path / "cap")

    assert report.verdict == "usable", (report.verdict, report.next_steps)
    assert report.camera_solve["kind"] == "nerf-transforms"
    assert report.camera_solve["registered_images"] == 36
    assert report.num_images == 36
    assert report.masks["count"] == 36


def test_the_inspector_over_reports_flash_pairs_on_a_capture_that_has_none(tmp_path):
    """A known false positive, recorded rather than fixed.

    This capture contains no flash / no-flash pairs at all -- every frame is
    lit. With no EXIF timestamps the detector falls back to brightness, and
    brightness here varies because the *light moves*, which is the same signal.
    It is advisory text and feeds no verdict, so it stays; but the real capture
    will have genuine pairs and the count must not be trusted as a total.
    """
    pytest.importorskip("PIL", reason="the inspector reads images through Pillow")
    from atlas.tools.inspect_capture import inspect_capture

    config = dataclasses.replace(SMALL, num_views=6, num_lights=6, width=64, height=64)
    generate_capture(tmp_path / "cap", config)
    report = inspect_capture(tmp_path / "cap")
    assert report.flash_no_flash_pairs is not None
    assert report.flash_no_flash_pairs > 0  # none exist; this is the false positive
