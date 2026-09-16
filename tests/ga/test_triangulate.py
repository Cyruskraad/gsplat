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
"""Triangulation by incidence.

Beyond exactness on noise-free data, these tests pin the two properties that
are easy to get quietly wrong: degenerate geometry must be *reported*, not
returned as a plausible number; and the estimator that minimizes reprojection
error must actually beat the algebraic ones when there is noise to separate
them.
"""

from __future__ import annotations

import pytest
import torch

from gsplat.contrib.ga.sfm import triangulate as tri
from tests.ga._helpers import synthetic_scene

DTYPE = torch.float64
SOLVERS = {
    "linear": tri.triangulate_linear,
    "midpoint": tri.triangulate_midpoint,
    "reprojection": tri.triangulate_reprojection,
}


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm(dim=-1).pow(2).mean().sqrt())


class TestExactRecovery:
    @pytest.mark.parametrize("name", list(SOLVERS))
    def test_recovers_noise_free_points(self, name):
        motors, intrinsics, world, pixels = synthetic_scene()
        got, valid = SOLVERS[name](motors, intrinsics, pixels)
        assert bool(valid.all())
        torch.testing.assert_close(got, world, atol=1e-7, rtol=0)

    @pytest.mark.parametrize("name", list(SOLVERS))
    def test_two_views_suffice(self, name):
        motors, intrinsics, world, pixels = synthetic_scene(views=2, points=64)
        got, valid = SOLVERS[name](motors, intrinsics, pixels)
        assert bool(valid.all())
        torch.testing.assert_close(got, world, atol=1e-7, rtol=0)


class TestDegenerateGeometry:
    def test_single_view_is_reported_invalid(self):
        """One ray does not determine a point; saying so beats guessing."""
        motors, intrinsics, _, pixels = synthetic_scene(views=1, points=32)
        for solver in (tri.triangulate_linear, tri.triangulate_midpoint):
            _, valid = solver(motors, intrinsics, pixels)
            assert not bool(valid.any())

    def test_masked_out_observations_do_not_contribute(self):
        """Zero-weighting a view must equal not passing that view at all."""
        motors, intrinsics, world, pixels = synthetic_scene(views=4, points=64)
        mask = torch.ones(4, 64, dtype=torch.bool)
        mask[3] = False
        masked, _ = tri.triangulate_linear(motors, intrinsics, pixels, mask)
        dropped, _ = tri.triangulate_linear(motors[:3], intrinsics[:3], pixels[:3])
        torch.testing.assert_close(masked, dropped, atol=1e-8, rtol=0)

    def test_a_point_with_one_remaining_view_is_invalid(self):
        motors, intrinsics, _, pixels = synthetic_scene(views=4, points=8)
        mask = torch.ones(4, 8, dtype=torch.bool)
        mask[1:, 0] = False  # point 0 keeps a single observation
        _, valid = tri.triangulate_linear(motors, intrinsics, pixels, mask)
        assert not bool(valid[0])
        assert bool(valid[1:].all())


class TestNoiseBehaviour:
    def test_reprojection_solver_beats_the_algebraic_ones(self):
        motors, intrinsics, world, pixels = synthetic_scene(points=2000)
        gen = torch.Generator().manual_seed(1)
        noisy = pixels + torch.randn(pixels.shape, generator=gen, dtype=DTYPE) * 2.0

        errors = {
            name: _rmse(solver(motors, intrinsics, noisy)[0], world)
            for name, solver in SOLVERS.items()
        }
        assert errors["reprojection"] < errors["linear"]
        assert errors["reprojection"] < errors["midpoint"]

    def test_reprojection_solver_minimizes_pixel_residual(self):
        motors, intrinsics, _, pixels = synthetic_scene(points=2000)
        gen = torch.Generator().manual_seed(2)
        noisy = pixels + torch.randn(pixels.shape, generator=gen, dtype=DTYPE) * 2.0

        def pixel_rmse(points):
            residual, valid = tri.reprojection_residuals(points, motors, intrinsics, noisy)
            return float(residual[valid].pow(2).sum(-1).mean().sqrt())

        best = pixel_rmse(tri.triangulate_reprojection(motors, intrinsics, noisy)[0])
        for other in ("linear", "midpoint"):
            assert best < pixel_rmse(SOLVERS[other](motors, intrinsics, noisy)[0])

    def test_error_scales_with_noise(self):
        motors, intrinsics, world, pixels = synthetic_scene(points=1000)
        gen = torch.Generator().manual_seed(3)
        previous = 0.0
        for sigma in (0.5, 1.0, 2.0, 4.0):
            noisy = pixels + torch.randn(pixels.shape, generator=gen, dtype=DTYPE) * sigma
            current = _rmse(tri.triangulate_reprojection(motors, intrinsics, noisy)[0], world)
            assert current > previous
            previous = current


class TestSolverIdentities:
    def test_midpoint_is_a_direct_solve(self):
        """Regression: the point-line residual is *linear* in the point, so the
        midpoint normal equations do not depend on the current estimate. An
        earlier version looped over this as if it were Gauss-Newton; the loop
        was provably dead work, and the name promised refinement it never did."""
        motors, intrinsics, _, pixels = synthetic_scene(points=128)
        gen = torch.Generator().manual_seed(4)
        noisy = pixels + torch.randn(pixels.shape, generator=gen, dtype=DTYPE) * 2.0
        once, _ = tri.triangulate_midpoint(motors, intrinsics, noisy)
        again, _ = tri.triangulate_midpoint(motors, intrinsics, noisy)
        torch.testing.assert_close(once, again, atol=0, rtol=0)

    def test_reprojection_solver_converges(self):
        motors, intrinsics, _, pixels = synthetic_scene(points=256)
        gen = torch.Generator().manual_seed(5)
        noisy = pixels + torch.randn(pixels.shape, generator=gen, dtype=DTYPE) * 2.0
        few, _ = tri.triangulate_reprojection(motors, intrinsics, noisy, iterations=3)
        many, _ = tri.triangulate_reprojection(motors, intrinsics, noisy, iterations=30)
        assert float((few - many).abs().max()) < 1e-6

    def test_triangulation_is_differentiable_in_the_cameras(self):
        motors, intrinsics, _, pixels = synthetic_scene(views=3, points=8)
        biv = torch.stack([torch.zeros(6, dtype=DTYPE) for _ in range(3)])
        biv.requires_grad_(True)
        from gsplat.contrib.ga import motor as mot

        perturbed = mot.motor_compose(mot.motor_exp(biv), motors)
        points, _ = tri.triangulate_linear(perturbed, intrinsics, pixels)
        points.pow(2).sum().backward()
        assert torch.isfinite(biv.grad).all() and float(biv.grad.abs().sum()) > 0
