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

"""The P1 gate: the two render paths are the same computation.

If any test in this file fails, the entire premise of ATLAS is wrong and no
amount of training will rescue it. They are deliberately free of any
rasteriser, so they run anywhere.
"""

import pytest
import torch

from gsplat.relight.functional import (
    composite,
    compositing_weights,
    contract,
    contract_screen,
    pack_transport,
    unpack_transport,
)

NUM_PIXELS = 7
NUM_PRIMITIVES = 23
NUM_ATOMS = 11


def _random_scene(dtype, seed=0):
    """Alphas, per-primitive transport and a shared light, all non-degenerate."""
    gen = torch.Generator().manual_seed(seed)
    alphas = (
        torch.rand(NUM_PIXELS, NUM_PRIMITIVES, generator=gen, dtype=dtype) * 0.6 + 0.05
    )
    transport = torch.randn(NUM_PRIMITIVES, 3, NUM_ATOMS, generator=gen, dtype=dtype)
    ell = torch.rand(3, NUM_ATOMS, generator=gen, dtype=dtype)
    return alphas, transport, ell


def _path_a(alphas, transport, ell):
    """Contract per primitive, then composite."""
    colors = contract(transport, ell)  # [P, 3]
    features = colors.unsqueeze(0).expand(alphas.shape[0], -1, -1)
    return composite(alphas, features)  # [pixels, 3]


def _path_b(alphas, transport, ell):
    """Composite the transport, then contract in screen space."""
    packed = pack_transport(transport)  # [P, 3B]
    features = packed.unsqueeze(0).expand(alphas.shape[0], -1, -1)
    composited = composite(alphas, features)  # [pixels, 3B]
    return contract_screen(unpack_transport(composited, NUM_ATOMS), ell)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_path_a_equals_path_b_in_float64(seed):
    """The linchpin, at a tolerance that leaves no room for a real difference."""
    alphas, transport, ell = _random_scene(torch.float64, seed)
    a = _path_a(alphas, transport, ell)
    b = _path_b(alphas, transport, ell)
    assert torch.max(torch.abs(a - b)) < 1e-12


def test_path_a_equals_path_b_meets_the_declared_float32_gate():
    """The gate as written in the design: max|A - B| < 1e-5 in float32."""
    alphas, transport, ell = _random_scene(torch.float32, seed=7)
    a = _path_a(alphas, transport, ell)
    b = _path_b(alphas, transport, ell)
    difference = float(torch.max(torch.abs(a - b)))
    # Both paths sum the same terms in different orders, so the residue is
    # rounding and nothing else. Assert it is far below the gate rather than
    # merely under it, so that a genuine algebraic divergence cannot hide
    # inside the tolerance.
    assert difference < 1e-5
    assert difference < 1e-6


def test_the_paths_differ_when_the_transport_is_perturbed():
    """A test that passes when both paths are broken identically is worthless.

    Perturbing one primitive's transport must move both paths, and must move
    them together. This pins that the comparison above is actually sensitive to
    the quantity it claims to compare.
    """
    alphas, transport, ell = _random_scene(torch.float64, seed=11)
    baseline = _path_a(alphas, transport, ell)
    perturbed = transport.clone()
    perturbed[3, 1, 2] += 0.5
    moved_a = _path_a(alphas, perturbed, ell)
    moved_b = _path_b(alphas, perturbed, ell)
    assert torch.max(torch.abs(moved_a - baseline)) > 1e-3
    assert torch.max(torch.abs(moved_a - moved_b)) < 1e-12


def test_superposition_holds_to_floating_point():
    """Linear in the illuminant by construction, not by penalty.

    A conditioned decoder has to be *taught* superposition with a loss term and
    retains a measurable error afterwards. Here the only residue is rounding,
    which is the difference between a structural property and a regularised
    one.
    """
    alphas, transport, _ = _random_scene(torch.float64, seed=5)
    gen = torch.Generator().manual_seed(99)
    ell1 = torch.rand(3, NUM_ATOMS, generator=gen, dtype=torch.float64)
    ell2 = torch.rand(3, NUM_ATOMS, generator=gen, dtype=torch.float64)

    combined = _path_a(alphas, transport, ell1 + ell2)
    separate = _path_a(alphas, transport, ell1) + _path_a(alphas, transport, ell2)
    assert torch.max(torch.abs(combined - separate)) < 1e-12

    combined_b = _path_b(alphas, transport, ell1 + ell2)
    separate_b = _path_b(alphas, transport, ell1) + _path_b(alphas, transport, ell2)
    assert torch.max(torch.abs(combined_b - separate_b)) < 1e-12


def test_scaling_the_light_scales_the_render():
    """Homogeneity, the other half of linearity."""
    alphas, transport, ell = _random_scene(torch.float64, seed=13)
    once = _path_a(alphas, transport, ell)
    thrice = _path_a(alphas, transport, 3.0 * ell)
    assert torch.max(torch.abs(thrice - 3.0 * once)) < 1e-12


def test_per_primitive_light_is_supported_by_path_a_only():
    """Near-field training gives every primitive its own light; Path A still works.

    Path B cannot represent this case -- there is no single ``ell`` to contract
    against in screen space -- and that limit is the honest boundary of the fast
    path, so it is pinned here rather than left to be discovered.
    """
    gen = torch.Generator().manual_seed(17)
    transport = torch.randn(
        NUM_PRIMITIVES, 3, NUM_ATOMS, generator=gen, dtype=torch.float64
    )
    per_primitive = torch.rand(
        NUM_PRIMITIVES, 3, NUM_ATOMS, generator=gen, dtype=torch.float64
    )
    colors = contract(transport, per_primitive)
    assert colors.shape == (NUM_PRIMITIVES, 3)
    expected = (transport * per_primitive).sum(dim=-1)
    assert torch.max(torch.abs(colors - expected)) == 0.0

    with pytest.raises(ValueError, match=r"ell must be \[3, B\]"):
        contract_screen(transport, per_primitive)


def test_compositing_weights_account_for_all_transmittance():
    """Weights sum to ``1 - prod(1 - alpha)``; the shortfall is the background."""
    alphas, _, _ = _random_scene(torch.float64, seed=3)
    weights = compositing_weights(alphas)
    expected = 1.0 - torch.prod(1.0 - alphas, dim=-1)
    assert torch.max(torch.abs(weights.sum(dim=-1) - expected)) < 1e-12


def test_opaque_front_primitive_hides_everything_behind_it():
    """Sanity on the compositing order: front to back, and alpha 1 occludes."""
    alphas = torch.tensor([[1.0, 0.9, 0.5]], dtype=torch.float64)
    features = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64
    )
    out = composite(alphas, features)
    assert torch.allclose(out, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))


def test_pack_unpack_is_an_exact_round_trip():
    _, transport, _ = _random_scene(torch.float64, seed=21)
    packed = pack_transport(transport)
    assert packed.shape == (NUM_PRIMITIVES, 3 * NUM_ATOMS)
    assert torch.equal(unpack_transport(packed, NUM_ATOMS), transport)


def test_pack_is_channel_major():
    """The splat channel order is part of the contract with the rasteriser."""
    transport = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    packed = pack_transport(transport)
    assert torch.equal(packed[0, :4], transport[0, 0])
    assert torch.equal(packed[0, 4:8], transport[0, 1])


# --- guards -----------------------------------------------------------------
# One test per guard, so that deleting any single check fails a named test.


def test_contract_rejects_transport_without_three_channels():
    with pytest.raises(ValueError, match=r"transport must be \[\.\.\., 3, B\]"):
        contract(torch.zeros(5, 2, 4), torch.zeros(2, 4))


def test_contract_rejects_mismatched_atom_count():
    with pytest.raises(ValueError, match="must match transport"):
        contract(torch.zeros(5, 3, 4), torch.zeros(3, 6))


def test_contract_rejects_per_primitive_ell_of_the_wrong_length():
    with pytest.raises(ValueError, match="rows but transport has"):
        contract(torch.zeros(5, 3, 4), torch.zeros(6, 3, 4))


def test_contract_rejects_ell_with_too_many_dimensions():
    with pytest.raises(ValueError, match=r"ell must be \[3, B\] or"):
        contract(torch.zeros(5, 3, 4), torch.zeros(2, 5, 3, 4))


def test_unpack_rejects_a_channel_count_that_is_not_three_times_the_atoms():
    with pytest.raises(ValueError, match="is not 3 \\* num_atoms"):
        unpack_transport(torch.zeros(5, 13), 4)


def test_unpack_rejects_a_non_positive_atom_count():
    with pytest.raises(ValueError, match="num_atoms must be >= 1"):
        unpack_transport(torch.zeros(5, 12), 0)


def test_composite_rejects_features_that_do_not_match_the_alphas():
    with pytest.raises(ValueError, match="must match"):
        composite(torch.zeros(4, 6), torch.zeros(4, 5, 3))
