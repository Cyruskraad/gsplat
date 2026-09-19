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

"""Classifying a capture by its own geometry, and splitting it accordingly.

Real data arrives with no flag saying whether the flash was on the camera, and
a synthetic capture's flag should be checked against its poses rather than
trusted. Everything here is measured from the viewmats and the light positions.

The numbers that set the thresholds: a hotshoe flash 5-20 cm from the lens, on
an object about a metre away, separates the light direction from the view
direction by 3-11 degrees. There is nothing to decouple at that scale, so the
view/light split degenerates and the pipeline uses interpolation against
extrapolation instead.
"""

import dataclasses
import json
import math

import pytest

torch = pytest.importorskip("torch")

from atlas.data.loader import (  # noqa: E402
    CO_LOCATED_DEGREES,
    DECOUPLED_DEGREES,
    EXTRAPOLATION_SET_NAMES,
    SET_NAMES,
    load_capture,
)
from atlas.data.synthetic import SyntheticConfig, generate_capture  # noqa: E402
from atlas.functional.splits import azimuthal_order, split_arc  # noqa: E402

ORBIT = SyntheticConfig(
    num_views=16, num_lights=16, num_primitives=24, width=12, height=12, num_atoms=6
)
CO_LOCATED = dataclasses.replace(
    ORBIT, flash_mode="bracket", flash_offset=(0.12, -0.04, 0.02)
)


def _tilted_orbit(num=24, tilt=0.4, wobble=0.2):
    """A handheld pass: circular, tilted out of the world planes, not level."""
    angles = torch.linspace(0, 2 * math.pi, num + 1)[:num]
    points = torch.stack(
        [torch.cos(angles), torch.sin(angles), wobble * torch.sin(3 * angles)], dim=-1
    )
    rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(tilt), -math.sin(tilt)],
            [0.0, math.sin(tilt), math.cos(tilt)],
        ]
    )
    return points @ rotation.T


# --- the azimuthal order ----------------------------------------------------


def test_the_orbit_axis_is_recovered_from_the_samples_not_assumed():
    """A handheld pass is never level, so assuming world +z would order a
    tilted orbit by its wobble instead of by its sweep."""
    order = azimuthal_order(_tilted_orbit())
    assert sorted(order.tolist()) == list(range(24))
    # Consecutive picks in the recovered order are neighbours on the circle.
    steps = [abs(order[i + 1] - order[i]) % 24 for i in range(23)]
    assert all(int(s) in (1, 23) for s in steps), steps


def test_a_vertical_orbit_is_ordered_too():
    """The case that a hard-coded world-xy plane gets wrong.

    A 23-degree tilt still projects onto xy as an ellipse, and atan2 on an
    ellipse is monotone, so the order survives an assumption it should not.
    An orbit in the yz plane projects onto xy as a *line*: every sample lands
    at angle 0 or pi and the ordering is destroyed. Shooting an object from
    above and below in a vertical arc is an ordinary thing to do.
    """
    points = _tilted_orbit(tilt=math.pi / 2, wobble=0.05)
    order = azimuthal_order(points)
    assert sorted(order.tolist()) == list(range(24))
    steps = [abs(order[i + 1] - order[i]) % 24 for i in range(23)]
    assert all(int(s) in (1, 23) for s in steps), order.tolist()


def test_the_order_survives_a_rotation_of_the_whole_capture():
    points = _tilted_orbit()
    spin = torch.tensor(
        [
            [math.cos(1.1), -math.sin(1.1), 0.0],
            [math.sin(1.1), math.cos(1.1), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    first = azimuthal_order(points).tolist()
    second = azimuthal_order(points @ spin.T).tolist()
    offset = second.index(first[0])
    assert second[offset:] + second[:offset] == first


def test_too_few_points_to_have_an_order_is_refused():
    with pytest.raises(ValueError, match="at least two points"):
        azimuthal_order(torch.zeros(1, 3))


# --- the arc ----------------------------------------------------------------


def test_the_arc_is_contiguous_in_angle_and_the_farthest_point_split_is_not():
    """The whole reason the arc exists: its hold-outs sit *outside* the trained
    region, which makes the score an extrapolation score."""
    from atlas.functional.splits import split_views

    points = _tilted_orbit()
    order = azimuthal_order(points).tolist()
    position = {index: rank for rank, index in enumerate(order)}

    arc = split_arc(points, 0.25)
    ranks = sorted(position[i] for i in arc.test.tolist())
    spans = [(ranks[i + 1] - ranks[i]) % 24 for i in range(len(ranks) - 1)]
    # One contiguous run: every step is 1 except at most one wrap.
    assert sum(1 for s in spans if s != 1) <= 1, ranks

    scattered = sorted(position[i] for i in split_views(points, 0, 6).test.tolist())
    gaps = [scattered[i + 1] - scattered[i] for i in range(len(scattered) - 1)]
    assert max(gaps) > 1, "the premise: farthest-point hold-outs are spread out"


def test_the_arc_and_the_rest_partition_the_capture():
    arc = split_arc(_tilted_orbit(), 0.3)
    assert set(arc.train.tolist()) | set(arc.test.tolist()) == set(range(24))
    assert not set(arc.train.tolist()) & set(arc.test.tolist())
    assert arc.val.numel() == 0


def test_a_different_start_gives_a_disjoint_arc():
    points = _tilted_orbit()
    first = set(split_arc(points, 0.2).test.tolist())
    second = set(split_arc(points, 0.2, start=12).test.tolist())
    assert not first & second


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5, -0.2])
def test_an_impossible_fraction_is_refused(fraction):
    with pytest.raises(ValueError, match=r"fraction must be in \(0, 1\)"):
        split_arc(_tilted_orbit(), fraction)


def test_holding_out_everything_is_refused():
    with pytest.raises(ValueError, match="every one of"):
        split_arc(_tilted_orbit(num=4), 0.99)


# --- classifying a real capture --------------------------------------------


def test_a_co_located_capture_is_recognised_and_its_offset_recovered(tmp_path):
    """Measured on the synthetic bracket capture: 4.6 degrees median against a
    planted offset of 12.9 cm, recovered as 13 cm."""
    generate_capture(tmp_path / "cap", CO_LOCATED, write_images=False)
    report = load_capture(tmp_path / "cap").coupling_report()

    assert report.verdict == "co_located"
    assert report.median_degrees < CO_LOCATED_DEGREES
    planted = math.sqrt(sum(c * c for c in CO_LOCATED.flash_offset))
    assert report.implied_offset_metres == pytest.approx(planted, rel=0.02)
    assert report.splits_are_independent is False
    assert report.num_shots == CO_LOCATED.num_views


def test_a_free_capture_is_recognised_as_decoupled(tmp_path):
    generate_capture(tmp_path / "cap", ORBIT, write_images=False)
    report = load_capture(tmp_path / "cap").coupling_report()
    assert report.verdict == "decoupled"
    assert report.median_degrees > DECOUPLED_DEGREES
    assert report.splits_are_independent is True


def test_the_report_says_what_to_do_about_it(tmp_path):
    generate_capture(tmp_path / "cap", CO_LOCATED, write_images=False)
    text = str(load_capture(tmp_path / "cap").coupling_report())
    assert "effectively on the lens" in text
    assert "interpolation against extrapolation" in text


def test_the_report_is_json_serialisable_for_provenance(tmp_path):
    generate_capture(tmp_path / "cap", CO_LOCATED, write_images=False)
    payload = load_capture(tmp_path / "cap").coupling_report().to_dict()
    json.dumps(payload)
    assert payload["verdict"] == "co_located"


# --- the split follows the measurement -------------------------------------


def test_a_co_located_capture_gets_the_extrapolation_scheme(tmp_path):
    generate_capture(tmp_path / "cap", CO_LOCATED, write_images=False)
    split = load_capture(tmp_path / "cap").split(num_test_views=2, num_test_lights=2)

    assert split.scheme == "extrapolation"
    assert split.names == EXTRAPOLATION_SET_NAMES
    counts = split.counts()
    assert counts["held_out_extrapolated"] > 0
    assert counts["held_out_interpolated"] > 0
    assert sum(counts.values()) == CO_LOCATED.num_views


def test_a_free_capture_still_gets_the_four_way_scheme(tmp_path):
    generate_capture(tmp_path / "cap", ORBIT, write_images=False)
    split = load_capture(tmp_path / "cap").split()
    assert split.scheme == "independent"
    assert split.names == SET_NAMES
    assert split.counts()["held_out_light"] > 0


def test_the_scheme_can_be_forced_either_way(tmp_path):
    generate_capture(tmp_path / "cap", ORBIT, write_images=False)
    capture = load_capture(tmp_path / "cap")
    forced = capture.split(scheme="extrapolation", path=tmp_path / "forced.json")
    assert forced.scheme == "extrapolation"


def test_an_unknown_scheme_lists_the_ones_that_exist(tmp_path):
    generate_capture(tmp_path / "cap", ORBIT, write_images=False)
    with pytest.raises(ValueError, match="'auto', 'independent' or 'extrapolation'"):
        load_capture(tmp_path / "cap").split(scheme="random")


def test_interpolated_frames_are_chosen_from_what_the_arc_left_behind(tmp_path):
    """An 'interpolated' frame has to be surrounded by training frames. Picking
    it before removing the arc would let it land on the lip of the hole."""
    generate_capture(tmp_path / "cap", CO_LOCATED, write_images=False)
    split = load_capture(tmp_path / "cap").split(num_test_views=2, num_test_lights=2)
    assert not set(split.held_out_interpolated) & set(split.held_out_extrapolated)


def test_the_scheme_round_trips_through_the_split_file(tmp_path):
    generate_capture(tmp_path / "cap", CO_LOCATED, write_images=False)
    first = load_capture(tmp_path / "cap").split()
    second = load_capture(tmp_path / "cap").split()
    assert second.scheme == first.scheme == "extrapolation"
    assert second.held_out_extrapolated == first.held_out_extrapolated
    assert second.counts() == first.counts()


def test_asking_a_split_for_a_set_its_scheme_does_not_have_says_which_it_has(tmp_path):
    generate_capture(tmp_path / "cap", CO_LOCATED, write_images=False)
    split = load_capture(tmp_path / "cap").split()
    with pytest.raises(KeyError, match="held_out_extrapolated"):
        split["nonsense"]
    # The four-way names still resolve, and are empty.
    assert split["held_out_light"] == ()
