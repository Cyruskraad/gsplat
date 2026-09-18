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

"""Inverse lighting: the capability the linear formulation buys.

No baseline that is non-linear in the illuminant can do this at all, so these
tests are as much a statement of what the design is for as a check that the
solver works.
"""

import pytest
import torch

from atlas.functional import normal_equations, solve_light

DTYPE = torch.float64
NUM_PIXELS = 400
NUM_ATOMS = 12


def _problem(seed=0, num_atoms=NUM_ATOMS):
    """Screen-space transport and the light that produced the target image."""
    gen = torch.Generator().manual_seed(seed)
    transport = torch.rand(NUM_PIXELS, 3, num_atoms, generator=gen, dtype=DTYPE)
    ell = torch.rand(3, num_atoms, generator=gen, dtype=DTYPE) * 2.0
    target = (transport * ell.unsqueeze(0)).sum(dim=-1)
    return transport, ell, target


def test_recovers_a_known_illumination():
    transport, ell, target = _problem(seed=1)
    solution = solve_light(transport, target, max_iterations=2000, tolerance=1e-14)
    assert torch.max(torch.abs(solution.ell - ell)) < 1e-4
    assert float(solution.residual) < 1e-8


def test_the_solve_is_independent_of_image_size_after_accumulation():
    """The pixels enter only through the normal equations.

    Doubling the pixel count by duplication must leave the recovered light
    unchanged -- that is what makes this a control rather than a batch job.
    """
    transport, _, target = _problem(seed=2)
    once = solve_light(transport, target, max_iterations=2000, tolerance=1e-14)
    twice = solve_light(
        torch.cat([transport, transport]),
        torch.cat([target, target]),
        max_iterations=2000,
        tolerance=1e-14,
    )
    assert torch.max(torch.abs(once.ell - twice.ell)) < 1e-6


def test_normal_equations_match_a_direct_accumulation():
    transport, _, target = _problem(seed=3)
    gram, rhs = normal_equations(transport, target)
    assert gram.shape == (3, NUM_ATOMS, NUM_ATOMS)
    assert rhs.shape == (3, NUM_ATOMS)
    for channel in range(3):
        design = transport[:, channel, :]  # [P, B]
        assert (
            torch.max(torch.abs(gram[channel] - design.transpose(0, 1) @ design))
            < 1e-10
        )
        assert (
            torch.max(
                torch.abs(rhs[channel] - design.transpose(0, 1) @ target[:, channel])
            )
            < 1e-10
        )


def test_weights_can_mask_pixels_out_entirely():
    transport, ell, target = _problem(seed=4)
    corrupted = target.clone()
    corrupted[NUM_PIXELS // 2 :] += 5.0
    weights = torch.ones(NUM_PIXELS, dtype=DTYPE)
    weights[NUM_PIXELS // 2 :] = 0.0
    solution = solve_light(
        transport, corrupted, weights=weights, max_iterations=4000, tolerance=1e-14
    )
    assert torch.max(torch.abs(solution.ell - ell)) < 1e-3


def test_non_negativity_binds_when_the_data_wants_a_negative_coefficient():
    """Without the constraint the solver invents negative radiance.

    That is not a cosmetic difference: a negative coefficient looks correct on
    the view it was fitted to and wrong from every other direction, which is
    the failure mode this constraint exists to prevent.
    """
    gen = torch.Generator().manual_seed(5)
    transport = torch.rand(NUM_PIXELS, 3, 4, generator=gen, dtype=DTYPE)
    signed = torch.tensor(
        [[1.0, -0.8, 0.5, 0.2], [0.3, -0.5, 1.2, 0.1], [0.9, -0.2, 0.4, 0.7]],
        dtype=DTYPE,
    )
    target = (transport * signed.unsqueeze(0)).sum(dim=-1)

    unconstrained = solve_light(transport, target, non_negative=False)
    assert float(unconstrained.ell.min()) < -1e-3

    constrained = solve_light(transport, target, max_iterations=4000)
    assert float(constrained.ell.min()) >= 0.0
    # The constrained fit is necessarily worse on this target, and must be.
    assert float(constrained.residual) > float(unconstrained.residual)


def test_unconstrained_solve_is_exact_on_a_consistent_system():
    transport, ell, target = _problem(seed=6)
    solution = solve_light(transport, target, non_negative=False, ridge=1e-12)
    assert torch.max(torch.abs(solution.ell - ell)) < 1e-6
    assert solution.iterations == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_solver_terminates_well_inside_its_iteration_budget(seed):
    """ "Interactive" is a claim about iteration count, so it is measured.

    Measured across these seeds: 149-159 iterations to a projected-gradient
    residual of 1e-10 relative, with a final coefficient error around 1e-8. The
    budget below is roughly 1.6x the worst observed, so a genuine convergence
    regression fails this rather than merely slowing it down.

    Before adaptive restart was added this did not converge at all within 500
    iterations, which is why the number is pinned and not left implicit.
    """
    transport, ell, target = _problem(seed=seed)
    solution = solve_light(transport, target, max_iterations=5000, tolerance=1e-10)
    assert solution.iterations < 250
    assert torch.max(torch.abs(solution.ell - ell)) < 1e-6


def test_ridge_keeps_an_unconstrained_direction_finite():
    """A view that constrains nothing about an atom must not blow the solve up.

    Zeroing one atom's column across every pixel makes the Gram matrix singular,
    which is the normal case rather than a pathological one: a single view never
    constrains the whole sphere.
    """
    transport, _, target = _problem(seed=8)
    transport[:, :, 3] = 0.0
    solution = solve_light(transport, target, max_iterations=1000)
    assert torch.isfinite(solution.ell).all()
    assert float(solution.ell[:, 3].abs().max()) < 1e-6


# --- guards -----------------------------------------------------------------


def test_normal_equations_reject_transport_without_three_channels():
    with pytest.raises(ValueError, match=r"transport must be \[P, 3, B\]"):
        normal_equations(
            torch.zeros(4, 2, 5, dtype=DTYPE), torch.zeros(4, 2, dtype=DTYPE)
        )


def test_normal_equations_reject_a_target_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"target must be \[P, 3\]"):
        normal_equations(
            torch.zeros(4, 3, 5, dtype=DTYPE), torch.zeros(5, 3, dtype=DTYPE)
        )


def test_normal_equations_reject_weights_of_the_wrong_length():
    with pytest.raises(ValueError, match=r"weights must be \[P\]"):
        normal_equations(
            torch.zeros(4, 3, 5, dtype=DTYPE),
            torch.zeros(4, 3, dtype=DTYPE),
            weights=torch.zeros(3, dtype=DTYPE),
        )


def test_normal_equations_reject_negative_weights():
    with pytest.raises(ValueError, match="weights must be non-negative"):
        normal_equations(
            torch.zeros(4, 3, 5, dtype=DTYPE),
            torch.zeros(4, 3, dtype=DTYPE),
            weights=-torch.ones(4, dtype=DTYPE),
        )


def test_solve_rejects_a_negative_ridge():
    transport, _, target = _problem(seed=9)
    with pytest.raises(ValueError, match="ridge must be >= 0"):
        solve_light(transport, target, ridge=-1.0)


def test_solve_rejects_a_zero_iteration_budget():
    transport, _, target = _problem(seed=10)
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        solve_light(transport, target, max_iterations=0)
