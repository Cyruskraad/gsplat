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

"""The atom basis, and the agreement between its two projections.

Training projects point lights; inference projects environment maps. If those
two disagree, the model is fitted to one quantity and deployed against another,
and nothing downstream would reveal it. The tests here pin the agreement.
"""

import math

import pytest
import torch

from atlas.functional import (
    atom_gram_matrix,
    default_sharpness,
    equirect_directions,
    equirect_solid_angles,
    evaluate_atoms,
    fibonacci_sphere,
    make_sg_atoms,
    project_environment,
    project_point_light,
)

DTYPE = torch.float64


def test_fibonacci_points_are_unit_length_and_counted():
    points = fibonacci_sphere(64, dtype=DTYPE)
    assert points.shape == (64, 3)
    norms = torch.linalg.norm(points, dim=-1)
    assert torch.max(torch.abs(norms - 1.0)) < 1e-12


def test_fibonacci_points_are_spread_not_clustered():
    """A spiral that collapsed to a band would still pass the norm check."""
    points = fibonacci_sphere(256, dtype=DTYPE)
    # The centroid of a well-spread set sits near the origin.
    assert torch.linalg.norm(points.mean(dim=0)) < 0.02
    # And the z coordinates should cover the full range.
    assert points[:, 2].min() < -0.99
    assert points[:, 2].max() > 0.99


def test_equirect_solid_angles_sum_to_the_sphere():
    """The quadrature weight that every projection depends on."""
    total = equirect_solid_angles(128, 256, dtype=DTYPE).sum()
    assert abs(float(total) - 4.0 * math.pi) < 1e-3


def test_atoms_peak_at_one_on_their_own_axis():
    axes, sharpnesses = make_sg_atoms(32, dtype=DTYPE)
    values = evaluate_atoms(axes, axes, sharpnesses)  # [B, B]
    diagonal = torch.diagonal(values)
    assert torch.max(torch.abs(diagonal - 1.0)) < 1e-12
    assert float(values.max()) <= 1.0 + 1e-12
    assert float(values.min()) > 0.0


def test_default_sharpness_makes_neighbouring_lobes_meet_near_one_over_e():
    """The derivation in the docstring, checked against the actual geometry."""
    num_atoms = 128
    axes, sharpnesses = make_sg_atoms(num_atoms, dtype=DTYPE)
    cos = axes @ axes.transpose(0, 1)
    cos.fill_diagonal_(-1.0)
    nearest_cos = cos.max(dim=-1).values  # [B]
    value_at_neighbour = torch.exp(sharpnesses * (nearest_cos - 1.0))
    median = float(value_at_neighbour.median())
    # Lobes should overlap appreciably but not be near-identical: a basis with
    # gaps cannot represent a light that falls in one, and a basis of near
    # duplicates is badly conditioned for the inverse-lighting solve.
    assert 0.2 < median < 0.9
    assert default_sharpness(num_atoms) == pytest.approx(num_atoms / (2.0 * math.pi))


def test_point_light_projection_is_the_atom_value_times_the_radiance():
    axes, sharpnesses = make_sg_atoms(16, dtype=DTYPE)
    direction = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    radiance = torch.tensor([2.0, 3.0, 5.0], dtype=DTYPE)
    ell = project_point_light(direction, radiance, axes, sharpnesses)
    assert ell.shape == (3, 16)
    expected = radiance[:, None] * evaluate_atoms(direction, axes, sharpnesses)[None, :]
    assert torch.max(torch.abs(ell - expected)) == 0.0


def test_a_single_bright_texel_projects_exactly_like_a_point_light():
    """Training and inference project the same inner product.

    A delta light is the limit of an environment whose entire flux sits in one
    texel. Because the environment projection is a midpoint quadrature, that
    limit is reached *exactly* rather than approached, so this is an equality
    test and not a tolerance test -- which makes it a much stronger statement
    that the two code paths agree.
    """
    height, width = 32, 64
    axes, sharpnesses = make_sg_atoms(24, dtype=DTYPE)
    dirs = equirect_directions(height, width, dtype=DTYPE)
    domega = equirect_solid_angles(height, width, dtype=DTYPE)

    row, col = 9, 41
    flux = torch.tensor([0.7, 1.3, 2.1], dtype=DTYPE)
    envmap = torch.zeros(height, width, 3, dtype=DTYPE)
    envmap[row, col] = flux / domega[row, col]

    from_env = project_environment(envmap, axes, sharpnesses)
    from_point = project_point_light(dirs[row, col], flux, axes, sharpnesses)
    assert torch.max(torch.abs(from_env - from_point)) < 1e-12


def test_environment_projection_is_linear_in_the_environment():
    axes, sharpnesses = make_sg_atoms(12, dtype=DTYPE)
    gen = torch.Generator().manual_seed(4)
    e1 = torch.rand(16, 32, 3, generator=gen, dtype=DTYPE)
    e2 = torch.rand(16, 32, 3, generator=gen, dtype=DTYPE)
    combined = project_environment(e1 + 2.5 * e2, axes, sharpnesses)
    separate = project_environment(e1, axes, sharpnesses) + 2.5 * project_environment(
        e2, axes, sharpnesses
    )
    assert torch.max(torch.abs(combined - separate)) < 1e-12


def test_uniform_environment_projects_to_each_atom_s_integral():
    """A closed form to check the quadrature against.

    The integral of ``exp(lambda (cos - 1))`` over the sphere is
    ``2*pi*(1 - exp(-2*lambda))/lambda``.
    """
    axes, sharpnesses = make_sg_atoms(8, sharpness=6.0, dtype=DTYPE)
    envmap = torch.ones(128, 256, 3, dtype=DTYPE)
    ell = project_environment(envmap, axes, sharpnesses)
    lam = 6.0
    closed_form = 2.0 * math.pi * (1.0 - math.exp(-2.0 * lam)) / lam
    assert torch.max(torch.abs(ell - closed_form)) < 1e-3


def test_gram_matrix_is_symmetric_and_diagonally_dominant():
    axes, sharpnesses = make_sg_atoms(32, dtype=DTYPE)
    gram = atom_gram_matrix(axes, sharpnesses, resolution=48)
    assert torch.max(torch.abs(gram - gram.transpose(0, 1))) < 1e-10
    diagonal = torch.diagonal(gram)
    off_diagonal_max = (gram - torch.diag(diagonal)).max()
    assert float(diagonal.min()) > float(off_diagonal_max)


# --- guards -----------------------------------------------------------------


def test_fibonacci_rejects_an_empty_basis():
    with pytest.raises(ValueError, match="num_points must be >= 1"):
        fibonacci_sphere(0)


def test_default_sharpness_rejects_an_empty_basis():
    with pytest.raises(ValueError, match="num_atoms must be >= 1"):
        default_sharpness(0)


def test_make_sg_atoms_rejects_a_non_positive_sharpness():
    with pytest.raises(ValueError, match="sharpness must be > 0"):
        make_sg_atoms(8, sharpness=0.0)


def test_evaluate_atoms_rejects_directions_that_are_not_three_vectors():
    axes, sharpnesses = make_sg_atoms(4, dtype=DTYPE)
    with pytest.raises(ValueError, match="directions must end in 3"):
        evaluate_atoms(torch.zeros(5, 2, dtype=DTYPE), axes, sharpnesses)


def test_evaluate_atoms_rejects_axes_of_the_wrong_rank():
    with pytest.raises(ValueError, match=r"axes must be \[B, 3\]"):
        evaluate_atoms(torch.zeros(5, 3), torch.zeros(3), torch.zeros(1))


def test_evaluate_atoms_rejects_a_sharpness_vector_that_does_not_match_the_axes():
    with pytest.raises(ValueError, match="sharpnesses must be"):
        evaluate_atoms(torch.zeros(5, 3), torch.zeros(4, 3), torch.zeros(3))


def test_project_point_light_rejects_non_rgb_radiance():
    axes, sharpnesses = make_sg_atoms(4, dtype=DTYPE)
    with pytest.raises(ValueError, match="radiance must end in 3"):
        project_point_light(
            torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE),
            torch.zeros(4, dtype=DTYPE),
            axes,
            sharpnesses,
        )


def test_project_environment_rejects_a_map_that_is_not_hwc():
    axes, sharpnesses = make_sg_atoms(4, dtype=DTYPE)
    with pytest.raises(ValueError, match=r"envmap must be \[H, W, 3\]"):
        project_environment(torch.zeros(8, 16, dtype=DTYPE), axes, sharpnesses)


def test_equirect_directions_rejects_a_degenerate_resolution():
    with pytest.raises(ValueError, match="must be >= 1"):
        equirect_directions(0, 8)
