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
"""Motor exponential, logarithm, and the sandwich product.

The primary oracle is an independent 4x4 matrix exponential (``tests/ga/_helpers.py``):
motors are *isomorphic* to SE(3), so "the GA path and the matrix path agree to
machine precision" is the correct bar. Anything less means a bug in the GA code,
not a difference of formulation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gsplat.contrib.ga import motor as mot
from tests.ga._helpers import MAX_THETA, random_bivectors, se3_matrix

DTYPE = torch.float64

#: Bivectors covering the cases where a naive screw implementation breaks:
#: no rotation, no translation, translation purely along/across the screw axis,
#: a rotation small enough to underflow the axis, and one at the branch edge.
EDGE_CASES = {
    "identity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "pure_rotation": [0.3, -0.1, 0.2, 0.0, 0.0, 0.0],
    "pure_translation": [0.0, 0.0, 0.0, 0.5, -0.4, 0.7],
    "pure_screw_parallel": [0.3, -0.1, 0.2, 0.21, -0.07, 0.14],
    "tiny_rotation": [1e-9, 2e-9, -1e-9, 0.5, -0.4, 0.7],
    "tiny_everything": [1e-11, 0.0, 0.0, 1e-11, 0.0, 0.0],
    "branch_edge": [0.0, 0.0, MAX_THETA * 0.999, 0.5, -0.4, 0.7],
    "large_translation": [0.1, 0.2, -0.3, 50.0, -80.0, 120.0],
}


def _biv(name: str) -> torch.Tensor:
    return torch.tensor(EDGE_CASES[name], dtype=DTYPE)


class TestExponential:
    @pytest.mark.parametrize("name", list(EDGE_CASES))
    def test_matches_matrix_exponential(self, name):
        """The motor sandwich must reproduce ``expm`` of the matching se(3) twist."""
        biv = _biv(name)
        motor = mot.motor_exp(biv)
        transform = se3_matrix(biv)

        pts = torch.tensor(
            np.random.default_rng(0).normal(size=(64, 3)), dtype=DTYPE
        )
        got = mot.motor_apply_point(motor.expand(64, 8), pts).numpy()
        want = pts.numpy() @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("name", list(EDGE_CASES))
    def test_is_a_unit_motor(self, name):
        """``M ~M == 1``: the defining condition for a rigid motion."""
        motor = mot.motor_exp(_biv(name))
        product = mot.motor_compose(motor, mot.motor_inverse(motor))
        torch.testing.assert_close(
            product, mot.motor_identity(dtype=DTYPE), atol=1e-12, rtol=0
        )

    def test_preserves_distances(self):
        # Distances are formed directly rather than with torch.cdist: cdist uses
        # a matmul expansion whose diagonal comes out at ~6e-8 instead of 0,
        # which would mask (or fake) a real rigidity failure at this tolerance.
        def pairwise(x):
            return (x[:, None, :] - x[None, :, :]).pow(2).sum(-1).sqrt()

        gen = torch.Generator().manual_seed(11)
        pts = torch.randn(32, 3, generator=gen, dtype=DTYPE)
        for biv in random_bivectors(8, gen):
            moved = mot.motor_apply_point(mot.motor_exp(biv).expand(32, 8), pts)
            torch.testing.assert_close(
                pairwise(moved), pairwise(pts), atol=1e-10, rtol=0
            )

    def test_pitch_from_pseudoscalar(self):
        """The pseudoscalar coefficient of a screw is ``pitch * sin(theta)``.

        ``motor_log`` relies on this to recover the screw-parallel translation,
        so the relation is pinned here rather than left implicit.
        """
        axis = torch.tensor([0.3, -0.1, 0.2], dtype=DTYPE)
        axis = axis / axis.norm()
        for theta in (0.2, 0.5, 1.0, 1.4):
            for pitch in (0.3, -0.8, 1.5):
                biv = torch.cat([axis * theta, axis * pitch])
                pseudo = mot.motor_exp(biv)[7]
                assert float(pseudo) == pytest.approx(pitch * math.sin(theta), abs=1e-12)


class TestLogarithm:
    @pytest.mark.parametrize("name", list(EDGE_CASES))
    def test_round_trip(self, name):
        biv = _biv(name)
        torch.testing.assert_close(
            mot.motor_log(mot.motor_exp(biv)), biv, atol=1e-9, rtol=0
        )

    def test_round_trip_randomized(self):
        gen = torch.Generator().manual_seed(5)
        biv = random_bivectors(2000, gen)
        torch.testing.assert_close(
            mot.motor_log(mot.motor_exp(biv)), biv, atol=1e-9, rtol=0
        )

    def test_canonicalizes_the_double_cover(self):
        """``M`` and ``-M`` are the same rigid motion, so they share a logarithm."""
        gen = torch.Generator().manual_seed(7)
        biv = random_bivectors(64, gen)
        motor = mot.motor_exp(biv)
        torch.testing.assert_close(
            mot.motor_log(motor), mot.motor_log(-motor), atol=1e-9, rtol=0
        )

    def test_half_turn_is_not_a_special_case(self):
        """A 180-degree rotation is where a ``cos(theta)`` pitch recovery divides by zero."""
        biv = torch.tensor([0.0, 0.0, math.pi / 2, 0.4, -0.3, 0.9], dtype=DTYPE)
        motor = mot.motor_exp(biv)
        assert abs(float(motor[0])) < 1e-15  # cos(theta) really is zero here
        recovered = mot.motor_log(motor)
        torch.testing.assert_close(mot.motor_exp(recovered), motor, atol=1e-9, rtol=0)


class TestComposition:
    def test_compose_matches_matrix_product(self):
        gen = torch.Generator().manual_seed(13)
        a, b = random_bivectors(16, gen), random_bivectors(16, gen)
        composed = mot.motor_compose(mot.motor_exp(a), mot.motor_exp(b))
        pts = torch.randn(16, 3, generator=gen, dtype=DTYPE)
        got = mot.motor_apply_point(composed, pts).numpy()
        want = np.stack(
            [
                (se3_matrix(a[i]) @ se3_matrix(b[i]) @ np.append(pts[i].numpy(), 1.0))[:3]
                for i in range(16)
            ]
        )
        np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)

    def test_normalize_is_a_no_op_on_valid_motors(self):
        gen = torch.Generator().manual_seed(17)
        motor = mot.motor_exp(random_bivectors(64, gen))
        torch.testing.assert_close(
            mot.motor_normalize(motor), motor, atol=1e-11, rtol=0
        )

    def test_normalize_repairs_drift(self):
        gen = torch.Generator().manual_seed(19)
        motor = mot.motor_exp(random_bivectors(64, gen))
        drifted = motor + 1e-3 * torch.randn(
            motor.shape, generator=gen, dtype=DTYPE
        )
        repaired = mot.motor_normalize(drifted)
        product = mot.motor_compose(repaired, mot.motor_inverse(repaired))
        torch.testing.assert_close(
            product, mot.motor_identity(64, dtype=DTYPE), atol=1e-10, rtol=0
        )


class TestSandwichActsOnEveryGrade:
    """The point of using motors: one operator transforms points, lines and planes."""

    def test_plane_transforms_consistently_with_its_points(self):
        gen = torch.Generator().manual_seed(23)
        biv = random_bivectors(1, gen)[0]
        motor = mot.motor_exp(biv)
        # plane z = 0, and points lying on it
        plane = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=DTYPE)
        pts = torch.randn(32, 3, generator=gen, dtype=DTYPE)
        pts[:, 2] = 0.0

        moved_plane = mot.motor_apply_plane(motor, plane)
        moved_pts = mot.motor_apply_point(motor.expand(32, 8), pts)
        residual = moved_pts @ moved_plane[:3] + moved_plane[3]
        torch.testing.assert_close(
            residual, torch.zeros(32, dtype=DTYPE), atol=1e-10, rtol=0
        )

    def test_line_transforms_consistently_with_its_points(self):
        from gsplat.contrib.ga import algebra as alg

        gen = torch.Generator().manual_seed(29)
        motor = mot.motor_exp(random_bivectors(1, gen)[0])
        a = torch.randn(3, generator=gen, dtype=DTYPE)
        b = torch.randn(3, generator=gen, dtype=DTYPE)
        like = torch.zeros((), dtype=DTYPE)

        line = alg.mv_to_line(alg.point_mv(a) & alg.point_mv(b), like=like)
        moved_line = mot.motor_apply_line(motor, line)
        line_of_moved = alg.mv_to_line(
            alg.point_mv(mot.motor_apply_point(motor, a))
            & alg.point_mv(mot.motor_apply_point(motor, b)),
            like=like,
        )
        torch.testing.assert_close(moved_line, line_of_moved, atol=1e-10, rtol=0)


class TestAutograd:
    """Gate 0: gradients must flow through the algebra backend."""

    def test_gradcheck_exp_and_apply(self):
        biv = torch.tensor(
            [0.3, -0.1, 0.2, 0.5, -0.4, 0.7], dtype=DTYPE, requires_grad=True
        )
        pts = torch.randn(4, 3, dtype=DTYPE)

        def fn(b):
            return mot.motor_apply_point(mot.motor_exp(b).expand(4, 8), pts)

        assert torch.autograd.gradcheck(fn, (biv,), eps=1e-6, atol=1e-7)

    def test_gradcheck_log(self):
        biv = torch.tensor([0.2, 0.1, -0.3, 0.4, 0.2, -0.1], dtype=DTYPE)
        motor = mot.motor_exp(biv).detach().requires_grad_(True)
        assert torch.autograd.gradcheck(mot.motor_log, (motor,), eps=1e-6, atol=1e-6)

    def test_gradcheck_compose(self):
        a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=DTYPE, requires_grad=True)
        b = torch.tensor([-0.2, 0.1, 0.05, 0.3, -0.1, 0.2], dtype=DTYPE, requires_grad=True)

        def fn(x, y):
            return mot.motor_compose(mot.motor_exp(x), mot.motor_exp(y))

        assert torch.autograd.gradcheck(fn, (a, b), eps=1e-6, atol=1e-7)

    def test_gradients_are_finite_at_the_identity(self):
        """``w = 0`` is where the unregularized screw split is 0/0."""
        biv = torch.zeros(6, dtype=DTYPE, requires_grad=True)
        pts = torch.randn(8, 3, dtype=DTYPE)
        mot.motor_apply_point(mot.motor_exp(biv).expand(8, 8), pts).pow(2).sum().backward()
        assert torch.isfinite(biv.grad).all()


class TestBatching:
    def test_leading_dimensions_are_preserved(self):
        gen = torch.Generator().manual_seed(31)
        biv = random_bivectors(24, gen).reshape(2, 3, 4, 6)
        motor = mot.motor_exp(biv)
        assert motor.shape == (2, 3, 4, 8)
        torch.testing.assert_close(mot.motor_log(motor), biv, atol=1e-9, rtol=0)

    def test_batched_matches_looped(self):
        gen = torch.Generator().manual_seed(37)
        biv = random_bivectors(16, gen)
        pts = torch.randn(16, 3, generator=gen, dtype=DTYPE)
        batched = mot.motor_apply_point(mot.motor_exp(biv), pts)
        looped = torch.stack(
            [mot.motor_apply_point(mot.motor_exp(biv[i]), pts[i]) for i in range(16)]
        )
        torch.testing.assert_close(batched, looped, atol=1e-12, rtol=0)
