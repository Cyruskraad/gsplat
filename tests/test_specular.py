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

"""View-dependent gloss, and the property it must not cost.

The first two tests are the whole point of the module. ATLAS exists because
radiance is linear in the illumination -- that is what makes relighting cost the
same whether the scene is lit by one flash or a full environment, and what makes
the two render paths exactly equal. A specular term is the obvious thing to
break it.

It does not, because the exactness theorem constrains the transport to be
independent of the **light**, not of the **view**. The prefiltered atom lookup
depends on the view and the primitive and nothing else, so it rides inside the
same contraction. If that ever stops being true these tests fail loudly, which
is the only acceptable way for it to stop being true.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from atlas.functional.atoms import equirect_directions, make_sg_atoms  # noqa: E402
from atlas.functional.specular import (  # noqa: E402
    DEFAULT_LADDER_HEIGHT,
    DEFAULT_ROUGHNESS_LEVELS,
    min_resolvable_roughness,
    build_roughness_ladder,
    direction_to_texel,
    reflect,
    sample_equirect,
    specular_weights,
)
from atlas.functional.transport import (  # noqa: E402
    composite,
    compositing_weights,
    contract,
    contract_screen,
    contract_weights,
)

ATOMS = 8


@pytest.fixture(scope="module")
def basis():
    axes, sharpnesses = make_sg_atoms(ATOMS)
    return axes.double(), sharpnesses.double()


@pytest.fixture(scope="module")
def ladder(basis):
    axes, sharpnesses = basis
    return build_roughness_ladder(axes, sharpnesses, height=64, width=128)


def _scene(num=24, seed=0):
    generator = torch.Generator().manual_seed(seed)
    normals = torch.nn.functional.normalize(
        torch.randn(num, 3, generator=generator, dtype=torch.float64), dim=-1
    )
    views = torch.nn.functional.normalize(
        torch.randn(num, 3, generator=generator, dtype=torch.float64), dim=-1
    )
    return (
        torch.randn(num, 3, ATOMS, generator=generator, dtype=torch.float64),
        torch.rand(num, 3, generator=generator, dtype=torch.float64),
        torch.rand(num, generator=generator, dtype=torch.float64) * 0.6 + 0.08,
        normals,
        views,
    )


# --- the property the project exists for -----------------------------------


def test_radiance_stays_linear_in_the_illumination_with_gloss_in_it(ladder):
    """The test this module lives or dies by.

    Measured: superposition and homogeneity residues of 1e-15 on a render whose
    specular term is a third of its magnitude. If a view-dependent term ever
    costs linearity in ``ell``, relighting stops being independent of how many
    lights the illumination contains -- which is the entire claim.
    """
    transport, albedo, roughness, normals, views = _scene()
    weights = specular_weights(ladder, reflect(views, normals), roughness)

    def render(ell):
        return contract(transport, ell) + albedo * contract_weights(weights, ell)

    generator = torch.Generator().manual_seed(1)
    first = torch.rand(3, ATOMS, generator=generator, dtype=torch.float64)
    second = torch.rand(3, ATOMS, generator=generator, dtype=torch.float64)

    additive = render(first + second) - (render(first) + render(second))
    homogeneous = render(2.5 * first) - 2.5 * render(first)
    assert float(additive.abs().max()) < 1e-13, float(additive.abs().max())
    assert float(homogeneous.abs().max()) < 1e-13

    # The premise: the specular term is a real part of this render, not noise.
    specular = albedo * contract_weights(weights, first)
    assert float(specular.abs().max()) > 0.2 * float(render(first).abs().max())


def test_the_two_render_paths_still_agree_when_the_transport_is_view_dependent(ladder):
    """The exactness theorem, re-checked with a view in the transport.

    The theorem needs ``M`` to be independent of the *light*; it says nothing
    about the view. So for a fixed camera the two groupings of the sum are
    still the same sum, and contract-then-composite equals
    composite-then-contract exactly as before.
    """
    transport, albedo, roughness, normals, views = _scene(num=12, seed=3)
    weights = specular_weights(ladder, reflect(views, normals), roughness)
    # The full view-dependent transport, as Path B would splat it.
    view_transport = transport + albedo.unsqueeze(-1) * weights.unsqueeze(1)

    generator = torch.Generator().manual_seed(4)
    alphas = torch.rand(12, generator=generator, dtype=torch.float64) * 0.4
    ell = torch.rand(3, ATOMS, generator=generator, dtype=torch.float64)

    # One pixel: `contract_screen` wants a leading spatial dimension, because
    # in use it is [H, W, 3, B].
    composited = (compositing_weights(alphas)[:, None, None] * view_transport).sum(0)
    path_a = composite(alphas, contract(view_transport, ell))
    path_b = contract_screen(composited.unsqueeze(0), ell)[0]
    assert float((path_a - path_b).abs().max()) < 1e-14, float(
        (path_a - path_b).abs().max()
    )


def test_the_specular_contraction_matches_the_equivalent_full_one(ladder):
    """``contract_weights`` is the cheap form of the same sum: one lobe shared
    across colours, so ``[N, B]`` instead of ``[N, 3, B]`` and no intermediate
    of that size anywhere."""
    _, albedo, roughness, normals, views = _scene(num=16, seed=5)
    weights = specular_weights(ladder, reflect(views, normals), roughness)
    ell = torch.rand(3, ATOMS, dtype=torch.float64)

    cheap = albedo * contract_weights(weights, ell)
    expanded = albedo.unsqueeze(-1) * weights.unsqueeze(1)  # [N, 3, B]
    assert torch.allclose(cheap, contract(expanded, ell), atol=1e-14)


def test_the_specular_contraction_takes_a_per_primitive_light_too():
    """Near-field training gives every primitive its own ``ell``."""
    weights = torch.rand(10, ATOMS, dtype=torch.float64)
    shared = torch.rand(3, ATOMS, dtype=torch.float64)
    per_primitive = shared.expand(10, 3, ATOMS).contiguous()
    assert torch.allclose(
        contract_weights(weights, shared),
        contract_weights(weights, per_primitive),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "ell,message",
    [
        (torch.zeros(3, ATOMS + 1), "must end in"),
        (torch.zeros(4, ATOMS), "must end in"),
        (torch.zeros(5, 3, ATOMS), "has 5 rows but weights has 10"),
    ],
)
def test_a_mismatched_light_is_refused(ell, message):
    with pytest.raises(ValueError, match=message):
        contract_weights(torch.zeros(10, ATOMS), ell.double())


# --- the equirect lookup, against the convention it must match -------------


def test_a_texel_direction_maps_back_to_its_own_texel():
    """``direction_to_texel`` is the inverse of ``equirect_directions``. Getting
    the half-texel offset wrong would shift every lookup by half a texel, which
    is invisible in an image and wrong everywhere."""
    height, width = 16, 32
    directions = equirect_directions(height, width, dtype=torch.float64)
    rows, columns = direction_to_texel(directions.reshape(-1, 3), height, width)
    expected_rows = torch.arange(height, dtype=torch.float64).repeat_interleave(width)
    expected_columns = torch.arange(width, dtype=torch.float64).repeat(height)
    assert torch.allclose(rows, expected_rows, atol=1e-9)
    assert torch.allclose(columns, expected_columns, atol=1e-9)


def test_sampling_a_texel_centre_returns_that_texel():
    generator = torch.Generator().manual_seed(6)
    maps = torch.rand(3, 8, 16, generator=generator, dtype=torch.float64)
    directions = equirect_directions(8, 16, dtype=torch.float64).reshape(-1, 3)
    sampled = sample_equirect(maps, directions)
    assert torch.allclose(sampled, maps.reshape(3, -1).T, atol=1e-9)


def test_azimuth_wraps_and_latitude_clamps():
    """Longitude is periodic and latitude is not: a direction past the pole is
    the pole, but a direction past 2*pi is a direction at 0."""
    maps = torch.arange(2 * 4 * 8, dtype=torch.float64).reshape(2, 4, 8)
    pole = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float64)
    assert torch.isfinite(sample_equirect(maps, pole)).all()

    just_under = torch.tensor(
        [[math.cos(-1e-9), math.sin(-1e-9), 0.0]], dtype=torch.float64
    )
    just_over = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(
        sample_equirect(maps, just_under), sample_equirect(maps, just_over), atol=1e-6
    )


def test_maps_of_the_wrong_rank_are_refused():
    with pytest.raises(ValueError, match=r"maps must be \[C, H, W\]"):
        sample_equirect(torch.zeros(4, 4), torch.zeros(2, 3))


# --- reflection -------------------------------------------------------------


def test_a_head_on_view_reflects_back_along_itself():
    normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(reflect(normal, normal), normal, atol=1e-12)


def test_a_grazing_view_reflects_across_the_normal():
    """Incoming at 45 degrees in x-z leaves at 45 degrees on the other side."""
    root = math.sqrt(0.5)
    view = torch.tensor([[root, 0.0, root]], dtype=torch.float64)
    normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    expected = torch.tensor([[-root, 0.0, root]], dtype=torch.float64)
    assert torch.allclose(reflect(view, normal), expected, atol=1e-12)


def test_reflection_preserves_the_angle_to_the_normal():
    generator = torch.Generator().manual_seed(7)
    views = torch.nn.functional.normalize(
        torch.randn(32, 3, generator=generator, dtype=torch.float64), dim=-1
    )
    normals = torch.nn.functional.normalize(
        torch.randn(32, 3, generator=generator, dtype=torch.float64), dim=-1
    )
    reflected = reflect(views, normals)
    assert torch.allclose(
        (views * normals).sum(-1), (reflected * normals).sum(-1), atol=1e-12
    )
    assert torch.allclose(reflected.norm(dim=-1), torch.ones(32, dtype=torch.float64))


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="must match"):
        reflect(torch.zeros(4, 3), torch.zeros(5, 3))


# --- the ladder -------------------------------------------------------------


def test_the_ladder_has_a_map_per_atom_per_level(basis):
    axes, sharpnesses = basis
    built = build_roughness_ladder(
        axes, sharpnesses, levels=(0.6, 0.9), height=16, width=32
    )
    assert built.maps.shape == (2, ATOMS, 16, 32)
    assert built.num_atoms == ATOMS
    assert built.levels.tolist() == [0.6, 0.9]


def test_a_rougher_level_is_a_broader_lobe(basis):
    """The whole point of the ladder: roughness band-limits. A sharp level must
    concentrate more of its energy near the atom's own axis than a rough one."""
    axes, sharpnesses = basis
    built = build_roughness_ladder(axes, sharpnesses, levels=(0.3, 1.0), height=64)
    peaks = built.maps.amax(dim=(-2, -1))  # [levels, atoms]
    assert bool((peaks[0] > peaks[1]).all()), peaks.tolist()


def test_a_lobe_too_sharp_for_the_map_is_refused_rather_than_silently_zero(basis):
    """The defect this guard exists for.

    A prefiltered map cannot hold a lobe narrower than one texel. Asking for
    roughness 0.05 at 32 rows means lambda = 320000, some four thousand times
    narrower, and the quadrature returns 5.6e-142 -- a silent collapse that
    renders as a missing highlight rather than as an error.
    """
    axes, sharpnesses = basis
    with pytest.raises(ValueError, match="narrower than one texel"):
        build_roughness_ladder(axes, sharpnesses, levels=(0.05, 0.4), height=32)


@pytest.mark.parametrize("height,expected", [(16, 0.527), (64, 0.263), (256, 0.132)])
def test_the_resolvable_roughness_follows_the_map_resolution(height, expected):
    """Measured. Mirror-like surfaces are out of range for any practical map
    size, which is honest: a co-located capture cannot determine them either."""
    assert min_resolvable_roughness(height) == pytest.approx(expected, abs=1e-3)


def test_every_default_level_is_resolvable_at_the_default_resolution():
    floor = min_resolvable_roughness(DEFAULT_LADDER_HEIGHT)
    assert min(DEFAULT_ROUGHNESS_LEVELS) >= floor, (DEFAULT_ROUGHNESS_LEVELS, floor)


@pytest.mark.parametrize(
    "levels,message",
    [
        ((), "at least one roughness level"),
        ((0.3, 0.1), "ascending"),
        ((0.0, 0.6), r"\(0, 1\]"),
        ((0.6, 1.7), r"\(0, 1\]"),
    ],
)
def test_a_malformed_ladder_is_refused(basis, levels, message):
    axes, sharpnesses = basis
    with pytest.raises(ValueError, match=message):
        build_roughness_ladder(axes, sharpnesses, levels=levels, height=16, width=32)


def test_the_default_levels_are_geometric(basis):
    """Equal ratios rather than equal steps: the prefiltered lobe changes far
    faster at the sharp end, so uniform spacing would waste levels where nothing
    happens and interpolate badly where everything does."""
    ratios = [
        DEFAULT_ROUGHNESS_LEVELS[i + 1] / DEFAULT_ROUGHNESS_LEVELS[i]
        for i in range(len(DEFAULT_ROUGHNESS_LEVELS) - 1)
    ]
    assert max(ratios) - min(ratios) < 0.15, ratios


# --- weights ----------------------------------------------------------------


def test_a_roughness_exactly_on_a_level_samples_that_level(ladder):
    directions = torch.nn.functional.normalize(
        torch.randn(
            9, 3, generator=torch.Generator().manual_seed(8), dtype=torch.float64
        ),
        dim=-1,
    )
    level = float(ladder.levels[2])
    weights = specular_weights(
        ladder, directions, torch.full((9,), level, dtype=torch.float64)
    )
    assert torch.allclose(
        weights, sample_equirect(ladder.maps[2], directions), atol=1e-12
    )


def test_a_roughness_between_levels_lands_between_them(ladder):
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    low, high = float(ladder.levels[1]), float(ladder.levels[2])
    middle = specular_weights(
        ladder, directions, torch.tensor([(low + high) / 2], dtype=torch.float64)
    )
    at_low = specular_weights(
        ladder, directions, torch.tensor([low], dtype=torch.float64)
    )
    at_high = specular_weights(
        ladder, directions, torch.tensor([high], dtype=torch.float64)
    )
    between = (middle - at_low) * (middle - at_high)
    assert bool((between <= 1e-15).all()), between


def test_a_roughness_outside_the_ladder_clamps_to_its_ends(ladder):
    directions = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    lowest, highest = float(ladder.levels[0]), float(ladder.levels[-1])
    for asked, expected in ((1e-6, lowest), (5.0, highest)):
        clamped = specular_weights(
            ladder, directions, torch.tensor([asked], dtype=torch.float64)
        )
        at_end = specular_weights(
            ladder, directions, torch.tensor([expected], dtype=torch.float64)
        )
        assert torch.allclose(clamped, at_end, atol=1e-12)


def test_a_roughness_of_the_wrong_length_names_the_length_it_wanted(ladder):
    with pytest.raises(ValueError, match=r"roughness must be \[4\]"):
        specular_weights(ladder, torch.zeros(4, 3).double(), torch.zeros(7).double())
