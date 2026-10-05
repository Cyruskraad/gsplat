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

"""Sizing the light basis by measurement.

The compression has to be *optimal*, not merely reasonable, because the whole
argument for choosing ``B`` from the singular-value spectrum rests on there
being no better ``B``-atom basis to choose instead.
"""

import pytest
import torch

from gsplat.relight.functional import (
    compress_transport,
    contract,
    energy_retained,
    project_light_to_compressed,
    rank_for_energy,
    transport_spectrum,
)

DTYPE = torch.float64
NUM_PRIMITIVES = 60
NUM_ATOMS = 16


def _low_rank_transport(true_rank, seed=0):
    """Transport with a known intrinsic rank, plus a little noise."""
    gen = torch.Generator().manual_seed(seed)
    left = torch.randn(NUM_PRIMITIVES * 3, true_rank, generator=gen, dtype=DTYPE)
    right = torch.randn(true_rank, NUM_ATOMS, generator=gen, dtype=DTYPE)
    matrix = left @ right
    return matrix.reshape(NUM_PRIMITIVES, 3, NUM_ATOMS)


def test_spectrum_reveals_a_planted_rank():
    """The plot that sizes ``B``, checked against a scene whose rank is known."""
    transport = _low_rank_transport(true_rank=5)
    spectrum = transport_spectrum(transport)
    assert spectrum.num_atoms == NUM_ATOMS
    values = spectrum.singular_values
    assert float(values[4]) > 1e-6
    assert float(values[5]) < 1e-8 * float(values[0])
    assert rank_for_energy(values, 0.999999) == 5


def test_full_rank_compression_is_lossless():
    transport = _low_rank_transport(true_rank=NUM_ATOMS, seed=1)
    compressed = compress_transport(transport, NUM_ATOMS)
    reconstructed = compressed.transport @ compressed.rotation
    assert torch.max(torch.abs(reconstructed - transport)) < 1e-10


def test_compression_is_optimal_against_alternative_bases():
    """Eckart--Young, checked rather than cited.

    A random rank-``r`` basis is fitted by least squares -- the best any such
    basis can do -- and the SVD truncation must still beat it. If this fails,
    picking ``B`` from the spectrum is picking it from the wrong quantity.
    """
    transport = _low_rank_transport(true_rank=12, seed=2)
    matrix = transport.reshape(-1, NUM_ATOMS)
    rank = 6
    compressed = compress_transport(transport, rank)
    svd_error = torch.linalg.matrix_norm(
        compressed.transport.reshape(-1, rank) @ compressed.rotation - matrix
    )

    gen = torch.Generator().manual_seed(3)
    for _ in range(8):
        basis = torch.randn(rank, NUM_ATOMS, generator=gen, dtype=DTYPE)
        # Best coefficients for this basis, i.e. the least-squares projection.
        coeffs = torch.linalg.lstsq(basis.transpose(0, 1), matrix.transpose(0, 1))
        alternative = (basis.transpose(0, 1) @ coeffs.solution).transpose(0, 1)
        alt_error = torch.linalg.matrix_norm(alternative - matrix)
        assert float(svd_error) <= float(alt_error) + 1e-10


def test_energy_retained_is_monotone_and_reaches_one():
    transport = _low_rank_transport(true_rank=9, seed=4)
    values = transport_spectrum(transport).singular_values
    fractions = [float(energy_retained(values, r)) for r in range(len(values) + 1)]
    assert fractions[0] == 0.0
    assert all(a <= b + 1e-15 for a, b in zip(fractions, fractions[1:]))
    assert fractions[-1] == pytest.approx(1.0)


def test_rank_for_energy_is_the_smallest_rank_meeting_the_threshold():
    transport = _low_rank_transport(true_rank=10, seed=5)
    values = transport_spectrum(transport).singular_values
    for threshold in (0.5, 0.9, 0.99):
        rank = rank_for_energy(values, threshold)
        assert float(energy_retained(values, rank)) >= threshold
        assert float(energy_retained(values, rank - 1)) < threshold


def test_compressed_transport_and_rotated_light_reproduce_the_radiance():
    """The runtime consequence: compressing costs one matvec on ``ell``.

    At full rank the compressed pair must give exactly the radiance the
    original pair gave -- otherwise the rotation and the coefficients disagree
    about which basis they are in, which is a silent error that would look like
    a colour shift after fine-tuning.
    """
    transport = _low_rank_transport(true_rank=NUM_ATOMS, seed=6)
    gen = torch.Generator().manual_seed(7)
    ell = torch.rand(3, NUM_ATOMS, generator=gen, dtype=DTYPE)

    reference = contract(transport, ell)
    compressed = compress_transport(transport, NUM_ATOMS)
    rotated = project_light_to_compressed(ell, compressed.rotation)
    assert rotated.shape == (3, NUM_ATOMS)
    assert (
        torch.max(torch.abs(contract(compressed.transport, rotated) - reference))
        < 1e-10
    )


def test_truncated_compression_degrades_gracefully_not_catastrophically():
    """Keeping the planted rank must cost essentially nothing."""
    transport = _low_rank_transport(true_rank=4, seed=8)
    gen = torch.Generator().manual_seed(9)
    ell = torch.rand(3, NUM_ATOMS, generator=gen, dtype=DTYPE)
    reference = contract(transport, ell)

    compressed = compress_transport(transport, 4)
    rotated = project_light_to_compressed(ell, compressed.rotation)
    error = torch.max(torch.abs(contract(compressed.transport, rotated) - reference))
    assert float(error) < 1e-10


# --- guards -----------------------------------------------------------------


def test_spectrum_rejects_transport_without_three_channels():
    with pytest.raises(ValueError, match=r"transport must be \[N, 3, B\]"):
        transport_spectrum(torch.zeros(4, 2, 8, dtype=DTYPE))


def test_compress_rejects_a_rank_above_the_atom_count():
    with pytest.raises(ValueError, match=r"rank must be in \[1, 8\]"):
        compress_transport(torch.zeros(4, 3, 8, dtype=DTYPE), 9)


def test_compress_rejects_a_zero_rank():
    with pytest.raises(ValueError, match=r"rank must be in \[1, 8\]"):
        compress_transport(torch.zeros(4, 3, 8, dtype=DTYPE), 0)


def test_energy_retained_rejects_a_rank_out_of_range():
    with pytest.raises(ValueError, match=r"rank must be in \[0, 3\]"):
        energy_retained(torch.ones(3, dtype=DTYPE), 4)


def test_rank_for_energy_rejects_a_threshold_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\]"):
        rank_for_energy(torch.ones(3, dtype=DTYPE), 0.0)


def test_projecting_light_rejects_a_rotation_built_for_other_atoms():
    with pytest.raises(ValueError, match="atoms but rotation expects"):
        project_light_to_compressed(
            torch.zeros(3, 8, dtype=DTYPE), torch.zeros(4, 6, dtype=DTYPE)
        )
