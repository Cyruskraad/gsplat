# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""The near-field model that the far-field claim rests on.

Every quantity here is analytic and every one of them is divided out before the
atoms see the observation. What these tests can establish is that the model is
the one it claims to be. What they cannot establish is that dividing it out
makes the transport transfer to a distant environment -- that needs an
environment probe and a photograph, and it is named as the principal risk in
``docs/relighting-atlas.md`` for exactly this reason.
"""

import math

import pytest
import torch

from gsplat.relight.functional import (
    cosine_power_profile,
    incident_radiance,
    inverse_square_falloff,
    light_directions,
    make_sg_atoms,
    project_point_light,
)

DTYPE = torch.float64


def test_directions_are_unit_length_and_point_at_the_light():
    means = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE)
    light = torch.tensor([0.0, 0.0, 2.0], dtype=DTYPE)
    dirs, distances = light_directions(means, light)
    assert torch.max(torch.abs(torch.linalg.norm(dirs, dim=-1) - 1.0)) < 1e-12
    assert torch.allclose(dirs[0], torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE))
    assert float(distances[0]) == pytest.approx(2.0)
    assert float(distances[1]) == pytest.approx(math.sqrt(5.0))


def test_every_primitive_sees_a_different_direction_to_a_near_light():
    """The reason near-field training cannot use a single shared ``ell``."""
    means = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE)
    light = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    dirs, _ = light_directions(means, light)
    assert float(torch.linalg.norm(dirs[0] - dirs[1])) > 0.5


def test_falloff_is_one_at_the_reference_distance_and_inverse_square_elsewhere():
    distances = torch.tensor([1.0, 2.0, 4.0], dtype=DTYPE)
    falloff = inverse_square_falloff(distances, reference_distance=1.0)
    assert torch.allclose(falloff, torch.tensor([1.0, 0.25, 0.0625], dtype=DTYPE))
    shifted = inverse_square_falloff(distances, reference_distance=2.0)
    assert float(shifted[1]) == pytest.approx(1.0)


def test_isotropic_profile_is_exactly_one_everywhere():
    dirs = torch.nn.functional.normalize(
        torch.randn(32, 3, generator=torch.Generator().manual_seed(1), dtype=DTYPE),
        dim=-1,
    )
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    profile = cosine_power_profile(dirs, axis, 0.0)
    assert torch.equal(profile, torch.ones(32, dtype=DTYPE))


def test_profile_peaks_on_axis_and_vanishes_behind_the_flash():
    axis = torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE)  # flash aimed at -z
    on_axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)  # surface below it
    behind = torch.tensor([[0.0, 0.0, -1.0]], dtype=DTYPE)
    assert float(cosine_power_profile(on_axis, axis, 4.0)[0]) == pytest.approx(1.0)
    assert float(cosine_power_profile(behind, axis, 4.0)[0]) == 0.0


def test_a_higher_exponent_narrows_the_beam():
    axis = torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE)
    oblique = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0, 1.0]], dtype=DTYPE), dim=-1
    )
    wide = float(cosine_power_profile(oblique, axis, 1.0)[0])
    narrow = float(cosine_power_profile(oblique, axis, 8.0)[0])
    assert 0.0 < narrow < wide < 1.0


def test_incident_radiance_combines_intensity_falloff_and_profile():
    means = torch.tensor([[0.0, 0.0, 0.0]], dtype=DTYPE)
    light = torch.tensor([0.0, 0.0, 2.0], dtype=DTYPE)
    intensity = torch.tensor([1.0, 2.0, 4.0], dtype=DTYPE)
    axis = torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE)
    dirs, radiance = incident_radiance(
        means, light, intensity, light_axis=axis, profile_exponent=2.0
    )
    assert torch.allclose(dirs[0], torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE))
    # On-axis profile is 1, falloff at 2 m is 1/4.
    assert torch.allclose(radiance[0], intensity * 0.25)


def test_near_field_observation_feeds_the_point_light_projection():
    """The join between this module and the atom basis, exercised end to end."""
    axes, sharpnesses = make_sg_atoms(16, dtype=DTYPE)
    means = torch.randn(7, 3, generator=torch.Generator().manual_seed(2), dtype=DTYPE)
    light = torch.tensor([0.0, 3.0, 0.0], dtype=DTYPE)
    intensity = torch.tensor([1.0, 1.0, 1.0], dtype=DTYPE)
    dirs, radiance = incident_radiance(means, light, intensity)
    ell = project_point_light(dirs, radiance, axes, sharpnesses)
    assert ell.shape == (7, 3, 16)
    # Per-primitive coefficients, which is exactly the shape Path A accepts.
    assert torch.isfinite(ell).all()


def test_distant_light_makes_the_directions_converge():
    """The far-field limit the environment projection assumes.

    Not a proof that transport transfers, only that the geometry degenerates
    the way the model says it does.
    """
    means = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=DTYPE)
    near, _ = light_directions(means, torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE))
    far, _ = light_directions(means, torch.tensor([0.0, 0.0, 1000.0], dtype=DTYPE))
    assert float(torch.linalg.norm(near[0] - near[1])) > 0.1
    assert float(torch.linalg.norm(far[0] - far[1])) < 1e-3


# --- guards -----------------------------------------------------------------


def test_light_directions_rejects_means_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"means must be \[N, 3\]"):
        light_directions(torch.zeros(4, 2, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))


def test_light_directions_rejects_a_light_position_that_is_not_a_three_vector():
    with pytest.raises(ValueError, match=r"light_position must be \[3\]"):
        light_directions(torch.zeros(4, 3, dtype=DTYPE), torch.zeros(4, dtype=DTYPE))


def test_falloff_rejects_a_non_positive_reference_distance():
    with pytest.raises(ValueError, match="reference_distance must be > 0"):
        inverse_square_falloff(torch.ones(3, dtype=DTYPE), reference_distance=0.0)


def test_falloff_rejects_non_positive_distances():
    with pytest.raises(ValueError, match="distances must be positive"):
        inverse_square_falloff(torch.zeros(3, dtype=DTYPE))


def test_profile_rejects_a_negative_exponent():
    with pytest.raises(ValueError, match="exponent must be >= 0"):
        cosine_power_profile(
            torch.zeros(2, 3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE), -1.0
        )


def test_profile_rejects_an_axis_that_is_not_a_three_vector():
    with pytest.raises(ValueError, match=r"light_axis must be \[3\]"):
        cosine_power_profile(
            torch.zeros(2, 3, dtype=DTYPE), torch.zeros(2, dtype=DTYPE), 2.0
        )


def test_incident_radiance_requires_an_axis_when_a_profile_is_requested():
    with pytest.raises(ValueError, match="light_axis is required"):
        incident_radiance(
            torch.zeros(2, 3, dtype=DTYPE),
            torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE),
            torch.ones(3, dtype=DTYPE),
            profile_exponent=2.0,
        )


def test_incident_radiance_rejects_a_non_rgb_intensity():
    with pytest.raises(ValueError, match=r"intensity must be \[3\]"):
        incident_radiance(
            torch.zeros(2, 3, dtype=DTYPE),
            torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE),
            torch.ones(4, dtype=DTYPE),
        )
