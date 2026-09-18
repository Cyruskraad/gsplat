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

"""The loader, and the split that the gate is measured against.

The split is the part worth testing hardest. It has to partition the capture
exactly, put the right frames in the right four sets, and -- once written --
refuse to become a different split, because two results measured against two
different cuts of the same data are not comparable and nothing else would say
so.
"""

import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from atlas.data import SET_NAMES, generate_capture, load_capture  # noqa: E402
from atlas.data.synthetic import SyntheticConfig  # noqa: E402

GRID = SyntheticConfig(
    num_views=6, num_lights=5, num_primitives=32, width=24, height=24, num_atoms=12
)


@pytest.fixture
def capture(tmp_path):
    generate_capture(tmp_path / "cap", GRID)
    return load_capture(tmp_path / "cap")


# --- what came back is what went out ---------------------------------------


def test_the_loader_recovers_the_grid_the_generator_wrote(capture):
    assert len(capture) == GRID.num_views * GRID.num_lights
    assert capture.view_indices == list(range(GRID.num_views))
    assert capture.light_indices == list(range(GRID.num_lights))
    assert capture.width == GRID.width and capture.height == GRID.height


def test_the_poses_survive_the_manifest_round_trip(tmp_path):
    written = generate_capture(tmp_path / "cap", GRID, write_images=False)
    loaded = load_capture(tmp_path / "cap")
    for frame in loaded:
        assert torch.allclose(
            frame.viewmat, written.viewmats[frame.view_index], atol=1e-12
        )
        assert torch.allclose(
            frame.light_position, written.light_positions[frame.light_index], atol=1e-12
        )


def test_the_intrinsics_survive_the_manifest_round_trip(tmp_path):
    written = generate_capture(tmp_path / "cap", GRID, write_images=False)
    loaded = load_capture(tmp_path / "cap")
    assert torch.allclose(loaded.intrinsics, written.intrinsics, atol=1e-12)


def test_an_image_comes_back_as_linear_radiance_on_the_scale_it_was_written(capture):
    image = capture.image(0)
    assert image.shape == (GRID.height, GRID.width, 3)
    assert image.dtype == torch.float64
    assert float(image.min()) >= 0.0
    # The premise: the object is actually lit in this frame.
    assert float(image.max()) > 0.05


def test_a_mask_comes_back_as_coverage_in_zero_to_one(capture):
    mask = capture.mask(0)
    assert mask.shape == (GRID.height, GRID.width)
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
    assert 0.05 < float((mask > 0.5).double().mean()) < 0.9


def test_exposure_is_divided_out_because_it_belongs_to_the_capture(tmp_path):
    """A per-shot exposure is a property of the photographer, not the object.
    Leaving it in the pixels would have the model learn it."""
    config = dataclasses.replace(GRID, exposure_jitter=0.4, num_views=3, num_lights=3)
    generate_capture(tmp_path / "cap", config)
    capture = load_capture(tmp_path / "cap")
    exposures = [f.exposure for f in capture]
    assert max(exposures) - min(exposures) > 0.05, exposures  # the premise

    manifest = json.loads((tmp_path / "cap" / "transforms.json").read_text())
    raw = capture.image(0, subtract_ambient=False)
    stored = manifest["frames"][0]["atlas"]["exposure"]
    from atlas.imageio import read_png

    written = read_png(capture.frames[0].image_path).double() / 255.0 * capture.scale
    assert torch.allclose(raw, (written / stored).clamp_min(0.0), atol=1e-12)


def test_ambient_is_subtracted_when_the_capture_has_some(tmp_path):
    config = dataclasses.replace(GRID, ambient=0.05, num_views=2, num_lights=2)
    generate_capture(tmp_path / "cap", config)
    capture = load_capture(tmp_path / "cap")
    assert capture.ambient == pytest.approx(0.05)
    with_ambient = capture.image(0, subtract_ambient=False)
    without = capture.image(0)
    assert float((with_ambient - without).min()) >= 0.0
    assert float(with_ambient.min()) > float(without.min())


def test_images_are_read_on_demand_and_not_at_load_time(tmp_path):
    """A 4k capture of a few hundred shots is tens of gigabytes. Constructing
    the dataset must not touch a single pixel."""
    generate_capture(tmp_path / "cap", GRID)
    for path in (tmp_path / "cap" / "images").iterdir():
        path.unlink()
    capture = load_capture(tmp_path / "cap")  # must still work
    assert len(capture) == GRID.num_views * GRID.num_lights
    with pytest.raises(FileNotFoundError):
        capture.image(0)


# --- the split --------------------------------------------------------------


def test_the_four_sets_partition_the_capture_exactly(capture):
    sets = [set(capture.split()[name]) for name in SET_NAMES]
    union = set().union(*sets)
    assert union == set(range(len(capture)))
    for i, first in enumerate(sets):
        for second in sets[i + 1 :]:
            assert not (first & second)


def test_the_counts_follow_the_grid_arithmetic(capture):
    """6 views and 5 lights, holding out 2 of each: 4x3 train, 2x3 held-out
    view, 4x2 held-out light, 2x2 both."""
    counts = capture.split(
        num_val_views=1, num_test_views=1, num_val_lights=1, num_test_lights=1
    ).counts()
    assert counts == {
        "train": 4 * 3,
        "held_out_view": 2 * 3,
        "held_out_light": 4 * 2,
        "held_out_both": 2 * 2,
    }


def test_each_set_holds_out_what_its_name_says(capture):
    split = capture.split()
    held_views = set(split.views.val.tolist()) | set(split.views.test.tolist())
    held_lights = set(split.lights.val.tolist()) | set(split.lights.test.tolist())

    def frames(name):
        return [capture.frames[i] for i in split[name]]

    assert all(
        f.view_index not in held_views and f.light_index not in held_lights
        for f in frames("train")
    )
    assert all(
        f.view_index in held_views and f.light_index not in held_lights
        for f in frames("held_out_view")
    )
    assert all(
        f.view_index not in held_views and f.light_index in held_lights
        for f in frames("held_out_light")
    )
    assert all(
        f.view_index in held_views and f.light_index in held_lights
        for f in frames("held_out_both")
    )


def test_the_split_is_written_once_and_read_back_thereafter(capture, tmp_path):
    first = capture.split(num_val_views=1, num_test_views=1)
    assert (capture.root / "split.json").is_file()
    # Different arguments, same answer: the file is authoritative.
    second = capture.split(num_val_views=2, num_test_views=2)
    assert first.counts() == second.counts()
    assert first.train == second.train


def test_a_split_from_a_different_manifest_is_refused_rather_than_reused(tmp_path):
    """Tuning a split is the easiest way to move a number without noticing."""
    generate_capture(tmp_path / "a", GRID)
    capture = load_capture(tmp_path / "a")
    capture.split()

    stored = json.loads((tmp_path / "a" / "split.json").read_text())
    stored["manifest_hash"] = "sha256:" + "0" * 64
    (tmp_path / "a" / "split.json").write_text(json.dumps(stored))

    with pytest.raises(ValueError, match="different manifest"):
        load_capture(tmp_path / "a").split()


def test_the_split_file_round_trips_through_json(capture):
    split = capture.split()
    reloaded = load_capture(capture.root).split()
    assert reloaded.to_dict() == split.to_dict()
    assert torch.equal(reloaded.views.test, split.views.test)


def test_asking_for_an_unknown_set_lists_the_four(capture):
    with pytest.raises(KeyError, match="held_out_both"):
        capture.split()["validation"]


# --- the finding, at the loader --------------------------------------------


def test_a_bracket_captures_held_out_light_frames_are_also_at_unseen_cameras(tmp_path):
    """The consequence of the protocol finding, stated as what is actually true.

    My first version of this asserted the two sets come back empty. They do
    not: the view split is farthest-point over camera *centres* and the light
    split is farthest-point over light *directions*, so even with the flash
    bolted to the camera the two selections need not pick the same indices, and
    frames land in both buckets.

    What is true is worse. In bracket mode frame *i* uses view *i* and light
    *i*, so if view *i* is held out then frame *i* is not in training and light
    *i* was therefore never seen either. Every held-out frame has an unseen
    camera **and** an unseen light, whichever bucket it landed in. The two
    numbers measure the same thing and their difference measures nothing about
    transport.
    """
    config = dataclasses.replace(GRID, flash_mode="bracket", num_views=8)
    generate_capture(tmp_path / "cap", config, write_images=False)
    capture = load_capture(tmp_path / "cap")
    assert capture.splits_are_independent is False

    split = capture.split()
    trained_views = {capture.frames[i].view_index for i in split.train}
    trained_lights = {capture.frames[i].light_index for i in split.train}

    # The premise: the buckets are not empty, so this is not vacuous.
    assert len(split.held_out_view) > 0 and len(split.held_out_light) > 0

    for name in ("held_out_view", "held_out_light", "held_out_both"):
        for index in split[name]:
            frame = capture.frames[index]
            assert frame.view_index not in trained_views, (name, index)
            assert frame.light_index not in trained_lights, (name, index)


def test_a_free_capture_shows_every_held_out_view_under_a_light_it_was_trained_on(
    tmp_path,
):
    """The property a bracket capture cannot have, and the reason the gate
    works at all: a held-out *view* is lit by a light the model has seen, so
    its score isolates view generalisation. Symmetrically for lights."""
    generate_capture(tmp_path / "cap", GRID, write_images=False)
    capture = load_capture(tmp_path / "cap")
    assert capture.splits_are_independent is True

    split = capture.split()
    trained_views = {capture.frames[i].view_index for i in split.train}
    trained_lights = {capture.frames[i].light_index for i in split.train}
    assert len(split.held_out_view) > 0 and len(split.held_out_light) > 0

    for index in split.held_out_view:
        assert capture.frames[index].light_index in trained_lights
    for index in split.held_out_light:
        assert capture.frames[index].view_index in trained_views


# --- refusals ---------------------------------------------------------------


def test_a_missing_manifest_says_what_to_run(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="inspect_capture"):
        load_capture(tmp_path / "empty")


def test_a_non_linear_colour_space_is_refused_rather_than_converted(tmp_path):
    """Converting here would hide where the conversion happened, and this
    project does no arithmetic on gamma-encoded values anywhere."""
    generate_capture(tmp_path / "cap", GRID, write_images=False)
    path = tmp_path / "cap" / "transforms.json"
    manifest = json.loads(path.read_text())
    manifest["atlas"]["colour_space"] = "srgb"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="only reads linear radiance"):
        load_capture(tmp_path / "cap")


def test_a_plain_nerf_manifest_without_light_positions_is_refused(tmp_path):
    generate_capture(tmp_path / "cap", GRID, write_images=False)
    path = tmp_path / "cap" / "transforms.json"
    manifest = json.loads(path.read_text())
    del manifest["frames"][0]["atlas"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="needs a light position per shot"):
        load_capture(tmp_path / "cap")


def test_a_manifest_missing_an_intrinsic_names_it(tmp_path):
    generate_capture(tmp_path / "cap", GRID, write_images=False)
    path = tmp_path / "cap" / "transforms.json"
    manifest = json.loads(path.read_text())
    del manifest["fl_x"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="'fl_x'"):
        load_capture(tmp_path / "cap")


def test_a_manifest_with_no_frames_is_refused(tmp_path):
    generate_capture(tmp_path / "cap", GRID, write_images=False)
    path = tmp_path / "cap" / "transforms.json"
    manifest = json.loads(path.read_text())
    manifest["frames"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lists no frames"):
        load_capture(tmp_path / "cap")
