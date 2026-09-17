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

"""The P1 prefiltering gate.

The claim is that prefiltering can be moved off the per-light-change path
entirely, because it commutes with the atom expansion. That is only worth
anything if the commutation is exact, so it is checked as an identity between
two independently computed quantities rather than as a plausible approximation.
"""

import pytest
import torch

from gsplat.relight.functional import (
    combine_prefiltered,
    evaluate_atoms,
    equirect_directions,
    make_sg_atoms,
    prefilter_atoms,
    prefilter_equirect,
    roughness_to_sharpness,
)

DTYPE = torch.float64


def test_prefiltering_is_linear_in_the_environment():
    gen = torch.Generator().manual_seed(2)
    e1 = torch.rand(16, 32, 3, generator=gen, dtype=DTYPE)
    e2 = torch.rand(16, 32, 3, generator=gen, dtype=DTYPE)
    combined = prefilter_equirect(e1 + 1.7 * e2, 20.0, out_height=8, out_width=16)
    separate = prefilter_equirect(
        e1, 20.0, out_height=8, out_width=16
    ) + 1.7 * prefilter_equirect(e2, 20.0, out_height=8, out_width=16)
    assert torch.max(torch.abs(combined - separate)) < 1e-12


def test_atom_space_prefiltering_matches_prefiltering_the_environment():
    """The gate: ``sum_k ell_k prefilter(A_k) == prefilter(sum_k ell_k A_k)``.

    The left side is what the runtime does -- a ``B``-term combination of maps
    already in memory. The right side is what it replaces -- a full convolution
    of the environment, per light change. They must be the same map.
    """
    num_atoms = 20
    source_h, source_w = 32, 64
    out_h, out_w = 12, 24
    kernel_sharpness = roughness_to_sharpness(0.35)

    axes, sharpnesses = make_sg_atoms(num_atoms, dtype=DTYPE)
    gen = torch.Generator().manual_seed(8)
    ell = torch.rand(3, num_atoms, generator=gen, dtype=DTYPE)

    # Runtime path: prefilter each atom once, then combine.
    atom_maps = prefilter_atoms(
        axes,
        sharpnesses,
        kernel_sharpness,
        height=out_h,
        width=out_w,
        source_height=source_h,
        source_width=source_w,
    )
    from_atoms = combine_prefiltered(atom_maps, ell)

    # Reference path: build the environment this ``ell`` describes, then
    # prefilter it directly.
    dirs = equirect_directions(source_h, source_w, dtype=DTYPE)
    atom_values = evaluate_atoms(dirs, axes, sharpnesses)  # [H, W, B]
    envmap = torch.einsum("hwb,cb->hwc", atom_values, ell)
    from_environment = prefilter_equirect(
        envmap, kernel_sharpness, out_height=out_h, out_width=out_w
    )

    assert from_atoms.shape == from_environment.shape == (out_h, out_w, 3)
    assert torch.max(torch.abs(from_atoms - from_environment)) < 1e-12


def test_prefiltering_a_constant_environment_returns_the_constant():
    """The kernel is normalised, so it must be a partition of unity."""
    envmap = torch.full((32, 64, 3), 0.375, dtype=DTYPE)
    out = prefilter_equirect(envmap, 8.0, out_height=8, out_width=16)
    assert torch.max(torch.abs(out - 0.375)) < 1e-12


def test_prefiltering_reduces_contrast():
    """A blur that does not blur would pass every linearity test above."""
    gen = torch.Generator().manual_seed(6)
    envmap = torch.rand(32, 64, 1, generator=gen, dtype=DTYPE)
    blurred = prefilter_equirect(envmap, 10.0, out_height=32, out_width=64)
    assert float(blurred.std()) < float(envmap.std())
    assert abs(float(blurred.mean()) - float(envmap.mean())) < 0.05


def test_a_sharper_kernel_blurs_less():
    gen = torch.Generator().manual_seed(10)
    envmap = torch.rand(32, 64, 1, generator=gen, dtype=DTYPE)
    soft = prefilter_equirect(envmap, 4.0, out_height=32, out_width=64)
    sharp = prefilter_equirect(envmap, 200.0, out_height=32, out_width=64)
    assert float(soft.std()) < float(sharp.std())


def test_roughness_maps_monotonically_to_sharpness():
    values = [roughness_to_sharpness(r) for r in (0.1, 0.3, 0.5, 0.8, 1.0)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(2.0)


# --- guards -----------------------------------------------------------------


def test_roughness_rejects_values_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"roughness must be in \(0, 1\]"):
        roughness_to_sharpness(0.0)
    with pytest.raises(ValueError, match=r"roughness must be in \(0, 1\]"):
        roughness_to_sharpness(1.5)


def test_prefilter_rejects_a_map_without_a_channel_axis():
    with pytest.raises(ValueError, match=r"envmap must be \[H, W, C\]"):
        prefilter_equirect(torch.zeros(8, 16, dtype=DTYPE), 4.0)


def test_prefilter_rejects_a_non_positive_sharpness():
    with pytest.raises(ValueError, match="sharpness must be > 0"):
        prefilter_equirect(torch.zeros(8, 16, 3, dtype=DTYPE), 0.0)


def test_combine_rejects_atom_maps_of_the_wrong_rank():
    with pytest.raises(ValueError, match=r"atom_maps must be \[B, H, W\]"):
        combine_prefiltered(
            torch.zeros(4, 8, dtype=DTYPE), torch.zeros(3, 4, dtype=DTYPE)
        )


def test_combine_rejects_a_light_vector_that_does_not_match_the_atoms():
    with pytest.raises(ValueError, match=r"ell must be \[3, 4\]"):
        combine_prefiltered(
            torch.zeros(4, 8, 16, dtype=DTYPE), torch.zeros(3, 5, dtype=DTYPE)
        )
